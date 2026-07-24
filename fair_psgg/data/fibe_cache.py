from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_fibe_cache_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        resolved = path
    else:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = (Path.cwd() / path, repo_root / path)
        resolved = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])

    if not resolved.is_file():
        raise FileNotFoundError(f"FIBE cache not found: {resolved}")
    return resolved.resolve()


class FIBEFeatureCache:
    """CPU-resident offline FIBE feature cache."""

    def __init__(self, path: str | Path, feature_dim: int = 21):
        self.path = resolve_fibe_cache_path(path)
        self.feature_dim = int(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")

        payload = _torch_load_cpu(self.path)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"FIBE cache must be a mapping, got {type(payload).__name__}: {self.path}"
            )

        mapping = payload.get("features_by_image", payload)
        if not isinstance(mapping, Mapping):
            raise TypeError(
                "FIBE cache field 'features_by_image' must be a mapping: "
                f"{self.path}"
            )

        self.features_by_image = mapping

    def _get_entry(self, image_id: int) -> Mapping[str, Any]:
        keys = (str(int(image_id)), int(image_id))
        for key in keys:
            if key in self.features_by_image:
                entry = self.features_by_image[key]
                if not isinstance(entry, Mapping):
                    raise TypeError(
                        f"FIBE cache entry for image {image_id} must be a mapping"
                    )
                return entry
        raise KeyError(
            f"Image {image_id} is absent from FIBE cache {self.path}"
        )

    def select_for_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select cached [subject, object, :] features for sampled relations.

        Returns:
            features: float32 tensor [R, feature_dim]
            valid: bool tensor [R]
        """
        pair_ids_local = (
            batch["sampled_relations"][:, :2].detach().cpu().to(torch.long)
        )
        image_ids = batch["image_id"].detach().cpu().to(torch.long)
        num_relations = batch["num_relations"].detach().cpu().to(torch.long)
        num_boxes = batch["num_boxes"].detach().cpu().to(torch.long)

        relation_count = int(pair_ids_local.shape[0])
        if int(num_relations.sum().item()) != relation_count:
            raise ValueError(
                "FIBE batch relation count mismatch: "
                f"sum(num_relations)={int(num_relations.sum().item())}, "
                f"sampled_relations={relation_count}"
            )

        relation_image_positions = torch.repeat_interleave(
            torch.arange(len(image_ids), dtype=torch.long),
            num_relations,
        )
        output_features = torch.zeros(
            (relation_count, self.feature_dim),
            dtype=torch.float32,
        )
        output_valid = torch.zeros(relation_count, dtype=torch.bool)

        for image_position, image_id_tensor in enumerate(image_ids):
            row_ids = torch.nonzero(
                relation_image_positions == image_position,
                as_tuple=False,
            ).flatten()
            if row_ids.numel() == 0:
                continue

            image_id = int(image_id_tensor.item())
            entry = self._get_entry(image_id)

            if "features" not in entry or "valid_pairs" not in entry:
                raise KeyError(
                    f"FIBE cache image {image_id} must contain "
                    "'features' and 'valid_pairs'"
                )

            features = torch.as_tensor(entry["features"], dtype=torch.float32)
            valid_pairs = torch.as_tensor(entry["valid_pairs"], dtype=torch.bool)

            if features.ndim != 3:
                raise ValueError(
                    f"Image {image_id}: expected features [N,N,D], "
                    f"got {tuple(features.shape)}"
                )
            if features.shape[0] != features.shape[1]:
                raise ValueError(
                    f"Image {image_id}: first two feature dimensions must match, "
                    f"got {tuple(features.shape)}"
                )
            if int(features.shape[2]) != self.feature_dim:
                raise ValueError(
                    f"Image {image_id}: expected feature_dim={self.feature_dim}, "
                    f"got {int(features.shape[2])}"
                )
            if tuple(valid_pairs.shape) != tuple(features.shape[:2]):
                raise ValueError(
                    f"Image {image_id}: valid_pairs shape "
                    f"{tuple(valid_pairs.shape)} does not match "
                    f"{tuple(features.shape[:2])}"
                )

            cached_num_objects = int(entry.get("num_objects", features.shape[0]))
            batch_num_objects = int(num_boxes[image_position].item())
            if cached_num_objects != int(features.shape[0]):
                raise ValueError(
                    f"Image {image_id}: num_objects={cached_num_objects}, "
                    f"feature matrix N={int(features.shape[0])}"
                )
            if cached_num_objects != batch_num_objects:
                raise ValueError(
                    f"Image {image_id}: cache num_objects={cached_num_objects}, "
                    f"batch num_boxes={batch_num_objects}"
                )

            local_pairs = pair_ids_local[row_ids]
            if local_pairs.numel() > 0:
                min_index = int(local_pairs.min().item())
                max_index = int(local_pairs.max().item())
                if min_index < 0 or max_index >= cached_num_objects:
                    raise IndexError(
                        f"Image {image_id}: pair index range "
                        f"[{min_index}, {max_index}] outside "
                        f"[0, {cached_num_objects - 1}]"
                    )

            selected = features[
                local_pairs[:, 0],
                local_pairs[:, 1],
            ].clone()
            selected_valid = valid_pairs[
                local_pairs[:, 0],
                local_pairs[:, 1],
            ].clone()

            if not torch.isfinite(selected).all():
                raise FloatingPointError(
                    f"Image {image_id}: selected FIBE features contain NaN/Inf"
                )

            # Invalid pairs must contribute exactly zero to the residual branch.
            selected[~selected_valid] = 0.0

            output_features[row_ids] = selected
            output_valid[row_ids] = selected_valid

        return output_features, output_valid


def build_fibe_cache(
    fibe_config: Any,
    split: str,
    override_path: str | Path | None = None,
) -> FIBEFeatureCache | None:
    if fibe_config is None or not bool(fibe_config.enabled):
        return None

    if override_path is not None:
        path_value = override_path
    elif split == "train":
        path_value = fibe_config.train_cache
    elif split in {"val", "validation"}:
        path_value = fibe_config.validation_cache
    elif split == "test":
        path_value = fibe_config.test_cache
    else:
        raise KeyError(f"Unsupported FIBE split: {split}")

    if path_value is None:
        raise ValueError(
            f"FIBE is enabled but no cache path is configured for split={split!r}"
        )

    return FIBEFeatureCache(
        path=path_value,
        feature_dim=int(fibe_config.feature_dim),
    )
