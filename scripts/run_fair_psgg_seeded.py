#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import random
import runpy
import sys
from pathlib import Path

# FLOODPSG_PROJECT_ROOT_V1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


seed = int(
    os.environ.get(
        "FLOODPSG_TRAIN_SEED",
        "3407",
    )
)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

runpy.run_module(
    "fair_psgg",
    run_name="__main__",
)
