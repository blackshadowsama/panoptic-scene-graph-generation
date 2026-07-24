#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fair_psgg.models.daniformer import DaniFormer


class TinyExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, image):
        return F.adaptive_avg_pool2d(self.conv(image), (16, 16))


def make_model(enabled):
    return DaniFormer(
        num_node_outputs=5,
        num_rel_outputs=9,
        extractor=TinyExtractor(),
        transformer_depth=1,
        embed_dim=384,
        patch_size=8,
        feature_shape=(16, 16, 16),
        use_semantics=False,
        use_masks=False,
        bg_ratio_strategy='sum',
        encode_coords=False,
        fibe_enabled=enabled,
        fibe_feature_dim=21,
        fibe_hidden_dim=64,
        fibe_bottleneck_dim=128,
        fibe_alpha_max=0.2,
        fibe_alpha_init=0.05,
        fibe_gate_bias_init=-3.0,
    )


def clone(data):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def main():
    torch.manual_seed(3407)
    base = make_model(False)
    d1 = make_model(True)
    result = d1.load_state_dict(base.state_dict(), strict=False)
    assert all(k.startswith('fibe_branch.') for k in result.missing_keys)
    assert not result.unexpected_keys
    base.eval()
    d1.eval()

    data = {
        'img': torch.randn(2, 3, 64, 64),
        'bboxes': torch.tensor([
            [2., 2., 12., 12.], [16., 3., 14., 18.], [5., 28., 20., 16.],
            [4., 5., 15., 20.], [24., 18., 18., 15.],
        ]),
        'num_boxes': torch.tensor([3, 2]),
        'pair_ids': torch.tensor([[0, 1], [2, 0], [4, 3]]),
        'box_categories': torch.tensor([0, 1, 2, 3, 4]),
    }
    fibe = torch.randn(3, 21)

    with torch.inference_mode():
        bs, bo, br = base(clone(data))
        invalid = clone(data)
        invalid['fibe_features'] = fibe.clone()
        invalid['fibe_valid'] = torch.zeros(3, dtype=torch.bool)
        ds0, do0, dr0 = d1(invalid)
    assert torch.equal(ds0, bs)
    assert torch.equal(do0, bo)
    assert torch.equal(dr0, br)

    with torch.inference_mode():
        valid = clone(data)
        valid['fibe_features'] = fibe.clone()
        valid['fibe_valid'] = torch.ones(3, dtype=torch.bool)
        ds1, do1, dr1 = d1(valid)

        chunked = clone(data)
        chunked['fibe_features'] = fibe.clone()
        chunked['fibe_valid'] = torch.ones(3, dtype=torch.bool)
        dsc, doc, drc = d1(chunked, max_relations=1)

    assert torch.equal(ds1, bs)
    assert torch.equal(do1, bo)
    delta = float((dr1 - br).abs().max())
    assert delta > 0.0
    assert torch.allclose(dsc, ds1, atol=1e-6, rtol=1e-6)
    assert torch.allclose(doc, do1, atol=1e-6, rtol=1e-6)
    assert torch.allclose(drc, dr1, atol=1e-6, rtol=1e-6)

    d1.train()
    d1.zero_grad(set_to_none=True)
    train = clone(data)
    train['fibe_features'] = fibe.clone()
    train['fibe_valid'] = torch.ones(3, dtype=torch.bool)
    _, _, rel = d1(train)
    rel.square().mean().backward()
    branch = d1.fibe_branch
    assert branch is not None
    assert branch.raw_alpha.grad is not None
    assert branch.encoder[0].weight.grad is not None
    assert branch.projection[0].weight.grad is not None
    assert branch.gate.weight.grad is not None

    print('PASS: full DaniFormer invalid-pair exact identity')
    print('PASS: FIBE changes relation head only')
    print('PASS: max_relations chunk alignment')
    print('PASS: full-model gradient flow')
    print(f'initial relation-logit max delta: {delta:.10f}')
    print('\nFIBE FULL DANIFORMER SMOKE: PASS')


if __name__ == '__main__':
    main()
