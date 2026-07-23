from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    PROJECT_ROOT / "data" / "floodpsg"
).resolve()

REVIEW_CSV = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_review_v1.csv"
)

VISUAL_DIR = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "boundary_audit_visuals_v1"
)

LABEL_DIR = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "boundary_audit_labels_v1"
)

BACKUP_CSV = (
    DATA_ROOT
    / "features"
    / "fibe_scalar_v1"
    / "fibe_boundary_audit_review_v1.before_ui_backup.csv"
)

AUDIT_LOG = (
    DATA_ROOT
    / "stats"
    / "dsformer_fibe_d1_v1"
    / "FIBE_BOUNDARY_AUDIT_UI_AUDIT_V1.jsonl"
)


MANUAL_FIELDS = [
    "manual_review_status",
    "manual_mask_pair_quality",
    "manual_geometry_relation",
    "manual_semantic_contact",
    "manual_apparent_boundary_quality",
    "manual_failure_mode",
    "manual_confidence",
    "manual_notes",
    "manual_saved_at",
    "manual_annotator",
    "manual_revision",
]


CHOICES = {
    "manual_review_status": {
        "pending",
        "complete",
        "exclude",
    },
    "manual_mask_pair_quality": {
        "valid",
        "target_mask_error",
        "hazard_mask_error",
        "both_mask_error",
        "ambiguous",
    },
    "manual_geometry_relation": {
        "overlap",
        "touching",
        "near_gap",
        "far_gap",
        "ambiguous",
    },
    "manual_semantic_contact": {
        "contact",
        "not_contact",
        "ambiguous",
    },
    "manual_apparent_boundary_quality": {
        "valid",
        "invalid",
        "not_applicable",
        "ambiguous",
    },
    "manual_confidence": {
        "high",
        "medium",
        "low",
    },
}


CLIENT_COLUMNS = [
    "audit_pair_id",
    "image_id",
    "global_image_key",
    "target_index",
    "hazard_index",
    "target_category",
    "hazard_category",
    "pair_family",
    *MANUAL_FIELDS,
]


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta
  name="viewport"
  content="width=device-width, initial-scale=1"
>
<title>FIBE 边界人工审查 V1</title>

<style>
:root {
  font-family:
    Arial,
    "Microsoft YaHei",
    sans-serif;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: #f1f3f5;
  color: #202124;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  background: #ffffff;
  border-bottom: 1px solid #c7c7c7;
  box-shadow: 0 1px 5px rgba(0,0,0,.12);
}

.topbar button,
.topbar input {
  min-height: 36px;
}

button {
  border: 1px solid #aeb4ba;
  background: white;
  border-radius: 5px;
  padding: 7px 12px;
  cursor: pointer;
  font-size: 14px;
}

button:hover {
  background: #eef3f8;
}

button.primary {
  background: #1667c5;
  color: white;
  border-color: #1667c5;
}

button.danger {
  background: #b3261e;
  color: white;
  border-color: #b3261e;
}

button.selected {
  background: #1769c2;
  color: white;
  border-color: #12539a;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.25);
}

button.exclude-selected {
  background: #b3261e;
  color: white;
  border-color: #7d1711;
}

.progress-wrap {
  flex: 1;
  min-width: 180px;
}

.progress-line {
  height: 10px;
  background: #d9dee3;
  border-radius: 5px;
  overflow: hidden;
}

.progress-value {
  height: 100%;
  width: 0;
  background: #1d7f3f;
}

.progress-text {
  margin-top: 3px;
  font-size: 13px;
  color: #555;
}

.layout {
  display: grid;
  grid-template-columns:
    minmax(620px, 1fr)
    430px;
  gap: 14px;
  padding: 14px;
  min-height: calc(100vh - 64px);
}

.viewer {
  background: white;
  border: 1px solid #c8cdd2;
  border-radius: 7px;
  padding: 10px;
  overflow: auto;
}

.viewer-header {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  margin-bottom: 8px;
  font-size: 15px;
}

.viewer-header strong {
  font-size: 19px;
}

.viewer img {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid #aeb4ba;
  background: white;
}

.form-panel {
  background: white;
  border: 1px solid #c8cdd2;
  border-radius: 7px;
  padding: 12px;
  overflow-y: auto;
}

.group {
  padding: 10px 0 14px;
  border-bottom: 1px solid #e0e0e0;
}

.group:last-child {
  border-bottom: 0;
}

.group-title {
  font-weight: 700;
  margin-bottom: 4px;
}

.group-help {
  color: #666;
  font-size: 12px;
  line-height: 1.45;
  margin-bottom: 8px;
}

.choice-group {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.choice-group button {
  font-size: 13px;
}

input[type="text"],
textarea,
select {
  width: 100%;
  border: 1px solid #aeb4ba;
  border-radius: 5px;
  padding: 8px;
  font-size: 14px;
}

textarea {
  min-height: 82px;
  resize: vertical;
}

.save-row {
  position: sticky;
  bottom: 0;
  display: flex;
  gap: 8px;
  margin: 12px -12px -12px;
  padding: 12px;
  background: white;
  border-top: 1px solid #c8cdd2;
}

.save-row button {
  flex: 1;
  min-height: 42px;
  font-weight: 700;
}

.message {
  min-height: 22px;
  margin-top: 8px;
  font-size: 13px;
}

.message.ok {
  color: #176b35;
}

.message.error {
  color: #b3261e;
}

.legend {
  padding: 8px;
  margin-top: 8px;
  background: #f6f8fa;
  border-radius: 5px;
  font-size: 12px;
  line-height: 1.55;
}

kbd {
  border: 1px solid #aaa;
  border-bottom-width: 2px;
  border-radius: 3px;
  background: white;
  padding: 1px 4px;
}

@media (max-width: 1100px) {
  .layout {
    grid-template-columns: 1fr;
  }

  .form-panel {
    overflow: visible;
  }
}
</style>
</head>

<body>
<div class="topbar">
  <button onclick="navigate(-1)">
    ← 上一条
  </button>

  <button onclick="navigate(1)">
    下一条 →
  </button>

  <button onclick="nextPending()">
    下一条未完成
  </button>

  <input
    id="jumpInput"
    type="text"
    placeholder="例如 28 或 FIBE-AUDIT-0028"
    style="width:220px"
  >

  <button onclick="jumpToPair()">
    跳转
  </button>

  <div class="progress-wrap">
    <div class="progress-line">
      <div
        id="progressValue"
        class="progress-value"
      ></div>
    </div>

    <div
      id="progressText"
      class="progress-text"
    ></div>
  </div>
</div>

<div class="layout">
  <main class="viewer">
    <div class="viewer-header">
      <strong id="pairId">加载中……</strong>
      <span id="positionText"></span>
      <span id="pairMeta"></span>
    </div>

    <a
      id="imageLink"
      href="#"
      target="_blank"
    >
      <img
        id="visualImage"
        alt="FIBE audit visualization"
      >
    </a>

    <div class="legend">
      红色为目标对象，蓝色为灾害对象。<br>
      几何关系只依据两个Mask；
      语义接触依据真实场景。<br>
      此页面不显示正负标签和谓词标签。
    </div>
  </main>

  <aside class="form-panel">
    <section class="group">
      <div class="group-title">
        1. Mask质量
      </div>

      <div class="group-help">
        红色和蓝色Mask是否分别对应正确对象。
      </div>

      <div
        class="choice-group"
        data-field="manual_mask_pair_quality"
      >
        <button data-value="valid">
          两者有效
        </button>
        <button data-value="target_mask_error">
          Target错误
        </button>
        <button data-value="hazard_mask_error">
          Hazard错误
        </button>
        <button data-value="both_mask_error">
          两者错误
        </button>
        <button data-value="ambiguous">
          无法判断
        </button>
      </div>
    </section>

    <section class="group">
      <div class="group-title">
        2. Mask几何关系
      </div>

      <div class="group-help">
        只判断独立Mask之间的图像空间关系。
      </div>

      <div
        class="choice-group"
        data-field="manual_geometry_relation"
      >
        <button data-value="overlap">
          面积重叠
        </button>
        <button data-value="touching">
          边界接触
        </button>
        <button data-value="near_gap">
          小间隙
        </button>
        <button data-value="far_gap">
          明显分离
        </button>
        <button data-value="ambiguous">
          无法判断
        </button>
      </div>
    </section>

    <section class="group">
      <div class="group-title">
        3. 真实场景语义接触
      </div>

      <div class="group-help">
        判断target是否真实接触、进入或受到hazard作用，
        不判断它是不是洪灾正样本。
      </div>

      <div
        class="choice-group"
        data-field="manual_semantic_contact"
      >
        <button data-value="contact">
          接触
        </button>
        <button data-value="not_contact">
          未接触
        </button>
        <button data-value="ambiguous">
          无法判断
        </button>
      </div>
    </section>

    <section class="group">
      <div class="group-title">
        4. 表观水体边界质量
      </div>

      <div class="group-help">
        water/river判断其局部上缘或接触边界是否可靠。
        barricade和mud_debris选择“不适用”。
      </div>

      <div
        class="choice-group"
        data-field="manual_apparent_boundary_quality"
      >
        <button data-value="valid">
          边界有效
        </button>
        <button data-value="invalid">
          边界无效
        </button>
        <button data-value="not_applicable">
          不适用
        </button>
        <button data-value="ambiguous">
          无法判断
        </button>
      </div>
    </section>

    <section class="group">
      <div class="group-title">
        5. 判断置信度
      </div>

      <div
        class="choice-group"
        data-field="manual_confidence"
      >
        <button data-value="high">
          高
        </button>
        <button data-value="medium">
          中
        </button>
        <button data-value="low">
          低
        </button>
      </div>
    </section>

    <section class="group">
      <div class="group-title">
        6. 样本状态
      </div>

      <div class="group-help">
        正常完成选择“完成”；Mask严重错误或无法使用选择“排除”。
      </div>

      <div
        class="choice-group"
        data-field="manual_review_status"
      >
        <button data-value="pending">
          暂存
        </button>
        <button data-value="complete">
          完成
        </button>
        <button data-value="exclude">
          排除
        </button>
      </div>
    </section>

    <section class="group">
      <div class="group-title">
        7. 排除或失败原因
      </div>

      <input
        id="failureMode"
        type="text"
        list="failureOptions"
        placeholder="仅排除样本需要填写"
      >

      <datalist id="failureOptions">
        <option value="target_mask_wrong_class">
        <option value="hazard_mask_wrong_class">
        <option value="both_masks_wrong">
        <option value="severe_mask_fragmentation">
        <option value="image_mask_misalignment">
        <option value="unresolvable_occlusion">
        <option value="semantic_scene_ambiguous">
        <option value="duplicate_or_invalid_pair">
      </datalist>
    </section>

    <section class="group">
      <div class="group-title">
        8. 备注
      </div>

      <textarea
        id="notes"
        placeholder="可选，例如：岸边拍摄、泳池场景、透视重叠等"
      ></textarea>
    </section>

    <div
      id="message"
      class="message"
    ></div>

    <div class="save-row">
      <button onclick="saveCurrent(false)">
        保存当前
      </button>

      <button
        class="primary"
        onclick="saveCurrent(true)"
      >
        保存并下一条
      </button>
    </div>
  </aside>
</div>

<script>
let items = [];
let currentIndex = 0;
let dirty = false;

const CATEGORY_ZH = {
  "person": "人员",
  "vehicle": "车辆",
  "road": "道路",
  "sidewalk": "人行道",
  "ground": "地面",
  "building": "建筑物",
  "underpass_bridge": "桥梁或地下通道",
  "drain": "排水口",
  "manhole": "检查井",
  "water": "积水或水体",
  "river": "河流",
  "mud_debris": "泥沙或碎屑",
  "barricade": "路障",
  "vegetation": "植被",
  "sign": "标志牌",
  "sky": "天空"
};

const FAMILY_ZH = {
  "human-water": "人员—水体",
  "vehicle-water": "车辆—水体",
  "road-surface-water": "道路表面—水体",
  "building-water": "建筑物—水体",
  "bridge-water": "桥梁或地下通道—水体",
  "drainage-water": "排水设施—水体",
  "road-debris": "道路—泥沙碎屑",
  "road-barricade": "道路—路障"
};

function zhCategory(value) {
  return CATEGORY_ZH[value] || value;
}

function zhFamily(value) {
  return FAMILY_ZH[value] || value;
}

const requiredCompleteFields = [
  "manual_mask_pair_quality",
  "manual_geometry_relation",
  "manual_semantic_contact",
  "manual_apparent_boundary_quality",
  "manual_confidence"
];

function normalizeStatus(value) {
  return value || "pending";
}

function currentItem() {
  return items[currentIndex];
}

function setMessage(text, type="") {
  const element = document.getElementById("message");
  element.textContent = text;
  element.className = "message " + type;
}

function progressCounts() {
  let complete = 0;
  let excluded = 0;

  for (const item of items) {
    const status = normalizeStatus(
      item.manual_review_status
    );

    if (status === "complete") {
      complete += 1;
    } else if (status === "exclude") {
      excluded += 1;
    }
  }

  return {
    complete,
    excluded,
    reviewed: complete + excluded,
    total: items.length
  };
}

function renderProgress() {
  const counts = progressCounts();
  const percent = counts.total
    ? 100 * counts.reviewed / counts.total
    : 0;

  document.getElementById(
    "progressValue"
  ).style.width = `${percent}%`;

  document.getElementById(
    "progressText"
  ).textContent =
    `已审查 ${counts.reviewed}/${counts.total} ` +
    `（完成 ${counts.complete}，排除 ${counts.excluded}）`;
}

function updateSelectedButtons() {
  const item = currentItem();

  document.querySelectorAll(
    ".choice-group"
  ).forEach(group => {
    const field = group.dataset.field;
    const value = normalizeStatus(
      field === "manual_review_status"
        ? item[field]
        : item[field] || ""
    );

    group.querySelectorAll(
      "button"
    ).forEach(button => {
      const selected =
        button.dataset.value === value;

      button.classList.toggle(
        "selected",
        selected
      );

      button.classList.toggle(
        "exclude-selected",
        selected &&
        field === "manual_review_status" &&
        value === "exclude"
      );
    });
  });
}

function renderCurrent() {
  const item = currentItem();

  if (!item) {
    return;
  }

  const visualUrl =
    `/visual/${item.audit_pair_id}.png`;

  document.getElementById(
    "pairId"
  ).textContent = item.audit_pair_id;

  document.getElementById(
    "positionText"
  ).textContent =
    `第 ${currentIndex + 1}/${items.length} 条`;

  document.getElementById(
    "pairMeta"
  ).textContent =
    `目标对象=${zhCategory(item.target_category)}，` +
    `灾害对象=${zhCategory(item.hazard_category)}，` +
    `关系族=${zhFamily(item.pair_family)}，` +
    `图像编号=${item.image_id}`;

  document.getElementById(
    "visualImage"
  ).src = visualUrl;

  document.getElementById(
    "imageLink"
  ).href = visualUrl;

  document.getElementById(
    "failureMode"
  ).value =
    item.manual_failure_mode || "";

  document.getElementById(
    "notes"
  ).value =
    item.manual_notes || "";

  if (
    !item.manual_apparent_boundary_quality &&
    ["barricade", "mud_debris"].includes(
      item.hazard_category
    )
  ) {
    item.manual_apparent_boundary_quality =
      "not_applicable";

    dirty = true;
  }

  updateSelectedButtons();
  renderProgress();
  setMessage("");
}

function setField(field, value) {
  const item = currentItem();
  item[field] = value;
  dirty = true;
  updateSelectedButtons();
}

document.querySelectorAll(
  ".choice-group"
).forEach(group => {
  group.querySelectorAll(
    "button"
  ).forEach(button => {
    button.addEventListener(
      "click",
      () => {
        setField(
          group.dataset.field,
          button.dataset.value
        );
      }
    );
  });
});

document.getElementById(
  "failureMode"
).addEventListener(
  "input",
  event => {
    currentItem().manual_failure_mode =
      event.target.value;
    dirty = true;
  }
);

document.getElementById(
  "notes"
).addEventListener(
  "input",
  event => {
    currentItem().manual_notes =
      event.target.value;
    dirty = true;
  }
);

function canNavigate() {
  if (!dirty) {
    return true;
  }

  return confirm(
    "当前标注尚未保存，确定离开吗？"
  );
}

function navigate(delta) {
  if (!canNavigate()) {
    return;
  }

  currentIndex = Math.max(
    0,
    Math.min(
      items.length - 1,
      currentIndex + delta
    )
  );

  dirty = false;
  renderCurrent();
}

function nextPending(force=false) {
  if (!force && !canNavigate()) {
    return;
  }

  for (
    let offset = 1;
    offset <= items.length;
    offset += 1
  ) {
    const index =
      (currentIndex + offset) % items.length;

    const status = normalizeStatus(
      items[index].manual_review_status
    );

    if (status === "pending") {
      currentIndex = index;
      dirty = false;
      renderCurrent();
      return;
    }
  }

  setMessage(
    "没有未完成样本。",
    "ok"
  );
}

function jumpToPair() {
  if (!canNavigate()) {
    return;
  }

  let text = document.getElementById(
    "jumpInput"
  ).value.trim();

  if (!text) {
    return;
  }

  if (/^\d+$/.test(text)) {
    text =
      "FIBE-AUDIT-" +
      text.padStart(4, "0");
  }

  const index = items.findIndex(
    item => item.audit_pair_id === text
  );

  if (index < 0) {
    setMessage(
      `未找到 ${text}`,
      "error"
    );
    return;
  }

  currentIndex = index;
  dirty = false;
  renderCurrent();
}

function annotationsForCurrent() {
  const item = currentItem();

  const result = {};

  [
    "manual_review_status",
    "manual_mask_pair_quality",
    "manual_geometry_relation",
    "manual_semantic_contact",
    "manual_apparent_boundary_quality",
    "manual_failure_mode",
    "manual_confidence",
    "manual_notes"
  ].forEach(field => {
    result[field] = item[field] || "";
  });

  result.manual_review_status =
    normalizeStatus(
      result.manual_review_status
    );

  return result;
}

function autoCompleteStatus(annotations) {
  if (
    annotations.manual_review_status !==
    "pending"
  ) {
    return;
  }

  const allFilled =
    requiredCompleteFields.every(
      field => Boolean(
        annotations[field]
      )
    );

  if (allFilled) {
    annotations.manual_review_status =
      "complete";

    currentItem().manual_review_status =
      "complete";

    updateSelectedButtons();
  }
}

async function saveCurrent(goNext) {
  const item = currentItem();
  const annotations =
    annotationsForCurrent();

  autoCompleteStatus(annotations);

  setMessage(
    "正在保存……"
  );

  try {
    const response = await fetch(
      "/api/save",
      {
        method: "POST",
        headers: {
          "Content-Type":
            "application/json"
        },
        body: JSON.stringify({
          audit_pair_id:
            item.audit_pair_id,
          annotations
        })
      }
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(
        result.error || "保存失败"
      );
    }

    Object.assign(
      item,
      result.item
    );

    dirty = false;
    renderProgress();

    setMessage(
      `已保存 ${item.audit_pair_id}`,
      "ok"
    );

    if (goNext) {
      setTimeout(
        () => nextPending(true),
        120
      );
    }
  } catch (error) {
    setMessage(
      error.message,
      "error"
    );
  }
}

async function loadItems() {
  try {
    const response = await fetch(
      "/api/items"
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(
        result.error || "加载失败"
      );
    }

    items = result.items;

    const firstPending = items.findIndex(
      item =>
        normalizeStatus(
          item.manual_review_status
        ) === "pending"
    );

    currentIndex =
      firstPending >= 0
        ? firstPending
        : 0;

    dirty = false;
    renderCurrent();
  } catch (error) {
    setMessage(
      error.message,
      "error"
    );
  }
}

document.addEventListener(
  "keydown",
  event => {
    const tag =
      document.activeElement.tagName;

    const editing =
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT";

    if (
      event.ctrlKey &&
      event.key === "Enter"
    ) {
      event.preventDefault();
      saveCurrent(true);
      return;
    }

    if (editing) {
      return;
    }

    if (event.key === "ArrowLeft") {
      event.preventDefault();
      navigate(-1);
    } else if (
      event.key === "ArrowRight"
    ) {
      event.preventDefault();
      navigate(1);
    }
  }
);

window.addEventListener(
  "beforeunload",
  event => {
    if (!dirty) {
      return;
    }

    event.preventDefault();
    event.returnValue = "";
  }
);

loadItems();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local browser-based annotation UI "
            "for the blinded FIBE boundary audit."
        )
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )
    parser.add_argument(
        "--annotator",
        default=os.environ.get(
            "USER",
            "unknown",
        ),
    )

    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


class ReviewStore:
    def __init__(
        self,
        *,
        review_csv: Path,
        visual_dir: Path,
        label_dir: Path,
        backup_csv: Path,
        audit_log: Path,
        annotator: str,
    ) -> None:
        self.review_csv = review_csv
        self.visual_dir = visual_dir
        self.label_dir = label_dir
        self.backup_csv = backup_csv
        self.audit_log = audit_log
        self.annotator = annotator
        self.lock = threading.RLock()

        self.frame = self._load()
        self.index_by_id = {
            str(row["audit_pair_id"]):
                int(index)
            for index, row
            in self.frame.iterrows()
        }

    def _load(self) -> pd.DataFrame:
        if not self.review_csv.is_file():
            raise FileNotFoundError(
                self.review_csv
            )

        if not self.visual_dir.is_dir():
            raise FileNotFoundError(
                self.visual_dir
            )

        frame = pd.read_csv(
            self.review_csv,
            encoding="utf-8-sig",
            keep_default_na=False,
            dtype=str,
        )

        if len(frame) != 240:
            raise RuntimeError(
                "Expected 240 review rows, "
                f"found {len(frame)}"
            )

        if (
            "audit_pair_id"
            not in frame.columns
        ):
            raise RuntimeError(
                "Review CSV has no "
                "audit_pair_id"
            )

        if frame[
            "audit_pair_id"
        ].duplicated().any():
            raise RuntimeError(
                "Duplicate audit_pair_id "
                "values"
            )

        forbidden = {
            "source_type",
            "predicate_name",
            "predicate_id",
        }

        leaked = (
            forbidden
            & set(frame.columns)
        )

        if leaked:
            raise RuntimeError(
                "Blinding violation: review "
                "CSV contains "
                f"{sorted(leaked)}"
            )

        for field in MANUAL_FIELDS:
            if field not in frame.columns:
                frame[field] = ""

        frame[
            "manual_review_status"
        ] = frame[
            "manual_review_status"
        ].map(
            lambda value: (
                clean_text(value)
                or "pending"
            )
        )

        frame[
            "manual_revision"
        ] = frame[
            "manual_revision"
        ].map(
            lambda value: (
                clean_text(value)
                or "0"
            )
        )

        if not self.backup_csv.exists():
            self.backup_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                self.review_csv,
                self.backup_csv,
            )

        self.label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.audit_log.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return frame

    def client_item(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.frame.loc[index]

        result = {}

        for column in CLIENT_COLUMNS:
            result[column] = clean_text(
                row.get(column, "")
            )

        return result

    def all_items(
        self,
    ) -> list[dict[str, Any]]:
        with self.lock:
            return [
                self.client_item(index)
                for index
                in self.frame.index
            ]

    def validate_annotations(
        self,
        annotations: dict[str, Any],
    ) -> dict[str, str]:
        allowed_input_fields = {
            "manual_review_status",
            "manual_mask_pair_quality",
            "manual_geometry_relation",
            "manual_semantic_contact",
            "manual_apparent_boundary_quality",
            "manual_failure_mode",
            "manual_confidence",
            "manual_notes",
        }

        unexpected = (
            set(annotations)
            - allowed_input_fields
        )

        if unexpected:
            raise ValueError(
                "Unexpected annotation fields: "
                f"{sorted(unexpected)}"
            )

        cleaned = {
            field: clean_text(
                annotations.get(
                    field,
                    "",
                )
            )
            for field
            in allowed_input_fields
        }

        status = (
            cleaned[
                "manual_review_status"
            ]
            or "pending"
        )

        cleaned[
            "manual_review_status"
        ] = status

        for field, allowed in (
            CHOICES.items()
        ):
            value = cleaned.get(
                field,
                "",
            )

            if (
                value
                and value not in allowed
            ):
                raise ValueError(
                    f"Invalid {field}: "
                    f"{value}"
                )

        if status == "complete":
            required = [
                "manual_mask_pair_quality",
                "manual_geometry_relation",
                "manual_semantic_contact",
                "manual_apparent_boundary_quality",
                "manual_confidence",
            ]

            missing = [
                field
                for field in required
                if not cleaned[field]
            ]

            if missing:
                raise ValueError(
                    "完成状态缺少字段："
                    + ", ".join(missing)
                )

        if (
            status == "exclude"
            and not cleaned[
                "manual_failure_mode"
            ]
        ):
            raise ValueError(
                "排除样本必须填写失败原因"
            )

        if len(
            cleaned["manual_notes"]
        ) > 4000:
            raise ValueError(
                "备注不能超过4000字符"
            )

        return cleaned

    def _write_csv(self) -> None:
        temporary = (
            self.review_csv.with_suffix(
                self.review_csv.suffix
                + ".tmp"
            )
        )

        self.frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
        )

        os.replace(
            temporary,
            self.review_csv,
        )

    def save(
        self,
        pair_id: str,
        annotations: dict[str, Any],
    ) -> dict[str, Any]:
        if pair_id not in self.index_by_id:
            raise KeyError(pair_id)

        cleaned = (
            self.validate_annotations(
                annotations
            )
        )

        with self.lock:
            index = self.index_by_id[
                pair_id
            ]

            old_row = self.client_item(
                index
            )

            try:
                old_revision = int(
                    clean_text(
                        self.frame.at[
                            index,
                            "manual_revision",
                        ]
                    )
                    or "0"
                )
            except ValueError:
                old_revision = 0

            timestamp = now_iso()

            for field, value in (
                cleaned.items()
            ):
                self.frame.at[
                    index,
                    field,
                ] = value

            self.frame.at[
                index,
                "manual_saved_at",
            ] = timestamp

            self.frame.at[
                index,
                "manual_annotator",
            ] = self.annotator

            self.frame.at[
                index,
                "manual_revision",
            ] = str(old_revision + 1)

            new_row = self.client_item(
                index
            )

            label_payload = {
                "version": (
                    "FIBE_BOUNDARY_"
                    "AUDIT_LABEL_V1"
                ),
                "audit_pair_id": pair_id,
                "saved_at": timestamp,
                "annotator": (
                    self.annotator
                ),
                "revision": (
                    old_revision + 1
                ),
                "image_id": (
                    new_row["image_id"]
                ),
                "global_image_key": (
                    new_row[
                        "global_image_key"
                    ]
                ),
                "target_index": (
                    new_row[
                        "target_index"
                    ]
                ),
                "hazard_index": (
                    new_row[
                        "hazard_index"
                    ]
                ),
                "target_category": (
                    new_row[
                        "target_category"
                    ]
                ),
                "hazard_category": (
                    new_row[
                        "hazard_category"
                    ]
                ),
                "pair_family": (
                    new_row["pair_family"]
                ),
                "annotations": {
                    field: new_row[field]
                    for field in (
                        "manual_review_status",
                        "manual_mask_pair_quality",
                        "manual_geometry_relation",
                        "manual_semantic_contact",
                        "manual_apparent_boundary_quality",
                        "manual_failure_mode",
                        "manual_confidence",
                        "manual_notes",
                    )
                },
            }

            label_path = (
                self.label_dir
                / f"{pair_id}.json"
            )

            atomic_write_json(
                label_path,
                label_payload,
            )

            self._write_csv()

            log_record = {
                "saved_at": timestamp,
                "annotator": (
                    self.annotator
                ),
                "audit_pair_id": pair_id,
                "revision": (
                    old_revision + 1
                ),
                "old": old_row,
                "new": new_row,
                "label_path": str(
                    label_path
                ),
            }

            with self.audit_log.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(
                        log_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            return new_row


app = Flask(__name__)
store: ReviewStore | None = None


def get_store() -> ReviewStore:
    if store is None:
        raise RuntimeError(
            "Review store is not initialized"
        )

    return store


@app.after_request
def disable_cache(response):
    response.headers[
        "Cache-Control"
    ] = (
        "no-store, no-cache, "
        "must-revalidate, max-age=0"
    )

    return response


@app.get("/")
def index():
    return render_template_string(
        HTML
    )


@app.get("/api/items")
def api_items():
    try:
        return jsonify(
            {
                "items": (
                    get_store().all_items()
                )
            }
        )
    except Exception as error:
        return jsonify(
            {"error": str(error)}
        ), 500


@app.post("/api/save")
def api_save():
    try:
        payload = request.get_json(
            force=True
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Request body must be JSON"
            )

        pair_id = clean_text(
            payload.get(
                "audit_pair_id"
            )
        )

        if not re.fullmatch(
            r"FIBE-AUDIT-\d{4}",
            pair_id,
        ):
            raise ValueError(
                "Invalid audit_pair_id"
            )

        annotations = payload.get(
            "annotations"
        )

        if not isinstance(
            annotations,
            dict,
        ):
            raise ValueError(
                "annotations must be "
                "an object"
            )

        item = get_store().save(
            pair_id,
            annotations,
        )

        return jsonify(
            {
                "ok": True,
                "item": item,
            }
        )

    except KeyError:
        return jsonify(
            {"error": "样本不存在"}
        ), 404

    except ValueError as error:
        return jsonify(
            {"error": str(error)}
        ), 400

    except Exception as error:
        return jsonify(
            {"error": str(error)}
        ), 500


@app.get(
    "/visual/<pair_id>.png"
)
def visual(pair_id: str):
    if not re.fullmatch(
        r"FIBE-AUDIT-\d{4}",
        pair_id,
    ):
        abort(404)

    filename = f"{pair_id}.png"

    path = (
        VISUAL_DIR / filename
    )

    if not path.is_file():
        abort(404)

    return send_from_directory(
        VISUAL_DIR,
        filename,
        mimetype="image/png",
    )


def main() -> None:
    global store

    args = parse_args()

    store = ReviewStore(
        review_csv=REVIEW_CSV,
        visual_dir=VISUAL_DIR,
        label_dir=LABEL_DIR,
        backup_csv=BACKUP_CSV,
        audit_log=AUDIT_LOG,
        annotator=args.annotator,
    )

    print("=" * 100)
    print(
        "FIBE BOUNDARY AUDIT UI V1"
    )
    print("=" * 100)
    print(
        "Review CSV:",
        REVIEW_CSV,
    )
    print(
        "Visual directory:",
        VISUAL_DIR,
    )
    print(
        "Per-pair labels:",
        LABEL_DIR,
    )
    print(
        "Backup CSV:",
        BACKUP_CSV,
    )
    print(
        "Audit log:",
        AUDIT_LOG,
    )
    print(
        "Annotator:",
        args.annotator,
    )
    print()
    print(
        f"Open: "
        f"http://localhost:{args.port}"
    )
    print(
        "Press Ctrl+C to stop."
    )

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
