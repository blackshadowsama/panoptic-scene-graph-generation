#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fair_psgg.data.fibe_cache import FIBEFeatureCache
from fair_psgg.data.split_batch import split_batch
from fair_psgg.trainer import prepare_batch

DATA = (ROOT / 'data' / 'floodpsg').resolve()
ANNO = DATA / 'annotations' / 'floodpsg_canonical_trainval_coarse8_groupstrict_v1.json'
TRAIN_CACHE = DATA / 'features' / 'fibe_scalar_v1' / 'train_features.pt'
VAL_CACHE = DATA / 'features' / 'fibe_scalar_v1' / 'validation_features.pt'


def records_from(payload):
    if isinstance(payload, list):
        return payload, {}
    for key in ('data', 'images', 'records'):
        if isinstance(payload.get(key), list):
            return payload[key], payload
    raise KeyError('records not found')


def select_records(records, val_ids, validation, count=2):
    out = []
    for record in records:
        is_val = int(record['image_id']) in val_ids
        if is_val == validation and record.get('relations'):
            out.append(record)
            if len(out) == count:
                return out
    raise RuntimeError('not enough relation-bearing records')


def make_batch(records):
    image_ids, num_boxes, num_relations = [], [], []
    categories, bboxes, relations = [], [], []
    for record in records:
        selected = record['relations'][:4]
        image_ids.append(int(record['image_id']))
        num_boxes.append(len(record['segments_info']))
        num_relations.append(len(selected))
        categories.extend(int(x['category_id']) for x in record['annotations'])
        bboxes.extend([float(v) for v in x['bbox']] for x in record['annotations'])
        relations.extend([int(r[0]), int(r[1]), 1] for r in selected)
    return {
        'image_id': torch.tensor(image_ids, dtype=torch.long),
        'num_boxes': torch.tensor(num_boxes, dtype=torch.long),
        'num_relations': torch.tensor(num_relations, dtype=torch.long),
        'sampled_relations': torch.tensor(relations, dtype=torch.long),
        'box_categories': torch.tensor(categories, dtype=torch.long),
        'bboxes': torch.tensor(bboxes, dtype=torch.float32),
        'img': torch.zeros(len(records), 3, 16, 16),
        'idx': torch.arange(len(records), dtype=torch.long),
    }


def audit(split, records, cache_path):
    cache = FIBEFeatureCache(cache_path, feature_dim=21)
    batch = make_batch(records)
    model_input, _, _, _ = prepare_batch(batch, torch.device('cpu'), fibe_cache=cache)
    local_pairs = batch['sampled_relations'][:, :2]
    rel_img_pos = torch.repeat_interleave(
        torch.arange(len(batch['image_id'])), batch['num_relations']
    )
    expected_f, expected_v = [], []
    for row, pos in enumerate(rel_img_pos.tolist()):
        image_id = int(batch['image_id'][pos])
        s, o = map(int, local_pairs[row].tolist())
        entry = cache._get_entry(image_id)
        valid = torch.as_tensor(entry['valid_pairs'][s, o], dtype=torch.bool)
        feat = torch.as_tensor(entry['features'][s, o], dtype=torch.float32).clone()
        if not bool(valid):
            feat.zero_()
        expected_f.append(feat)
        expected_v.append(valid)
    expected_f = torch.stack(expected_f)
    expected_v = torch.stack(expected_v)
    assert torch.equal(model_input['fibe_features'], expected_f)
    assert torch.equal(model_input['fibe_valid'], expected_v)
    assert bool(expected_v.all()), f'{split}: GT relation mapped to invalid FIBE pair'

    offsets = torch.cat((torch.tensor([0]), batch['num_boxes'][:-1])).cumsum(0)
    expected_global = local_pairs + torch.repeat_interleave(
        offsets, batch['num_relations']
    )[:, None]
    assert torch.equal(model_input['pair_ids'], expected_global)

    chunks_f, chunks_v = [], []
    for sub in split_batch(batch, max_relations=3):
        sub_input, _, _, _ = prepare_batch(
            sub, torch.device('cpu'), fibe_cache=cache
        )
        chunks_f.append(sub_input['fibe_features'])
        chunks_v.append(sub_input['fibe_valid'])
    assert torch.equal(torch.cat(chunks_f), expected_f)
    assert torch.equal(torch.cat(chunks_v), expected_v)

    print(f'PASS: {split} real-cache prepare_batch alignment')
    print('  image_ids:', batch['image_id'].tolist())
    print('  relations:', len(expected_f))
    print('  global_pair_ids:', expected_global.tolist())


def main():
    for path in (ANNO, TRAIN_CACHE, VAL_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = json.loads(ANNO.read_text(encoding='utf-8'))
    records, root = records_from(payload)
    val_ids = {int(x) for x in root.get('test_image_ids', [])}
    audit('train', select_records(records, val_ids, False), TRAIN_CACHE)
    audit('validation', select_records(records, val_ids, True), VAL_CACHE)
    print('\nFIBE REAL-CACHE PREPARE_BATCH SMOKE: PASS')


if __name__ == '__main__':
    main()
