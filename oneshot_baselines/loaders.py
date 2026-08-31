"""Loaders for the repo's native artifact formats.

Formats (verified against the released codebase
and the format-mapping notes):

- Dataset CSV (data/enterprise_ticket_dataset_1000.csv): column ``ticket_text``
  plus 24 ground-truth label columns (POV_COLUMNS below, exact order from
  dataset_generation/presets/enterprise_ticket/spec.py CSV_COLUMNS).
- Benchmark spec JSON: keys dataset_name / items / themes / references, where
  each reference has ``assignment`` = {item_text: cluster_id}.
- Batch-run dir (data/batch_run_results/<name>/):
  results.jsonl manifest; per run <config_slug>/<run_id>/clustering.json with
  {"clusters": [{"name": str, "members": [item_text, ...]}]}, plus optional
  selected_topic.json {"topic_id": ...} and config metadata keys carrying the
  run's POV.
- run_partitions.csv (long format from the export sheet):
  run_id, pov_name, item_id, cluster_id.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

POV_COLUMNS = [
    "department",
    "sub_department",
    "requester_role",
    "owning_team",
    "ticket_category",
    "issue_type",
    "intent_group",
    "sub_intent",
    "intent_label_coarse",
    "intent_label_normal",
    "intent_label_fine",
    "organizational_label_coarse",
    "organizational_label_normal",
    "organizational_label_fine",
    "system_name",
    "business_event_context",
    "workflow_stage",
    "approval_required",
    "dependency_type",
    "urgency",
    "severity",
    "impact_scope",
    "writing_style",
    "clarity_level",
]

TEXT_COLUMN = "ticket_text"

# The 12 themes in the shipped default benchmark spec
# (DEFAULT_ENTERPRISE_TICKET_THEME_DEFINITIONS in benchmark_builders.py).
DEFAULT_SPEC_THEMES = [
    "department",
    "ticket_category",
    "intent_group",
    "issue_type",
    "sub_intent",
    "system_name",
    "organizational_label_coarse",
    "organizational_label_normal",
    "organizational_label_fine",
    "intent_label_coarse",
    "intent_label_normal",
    "intent_label_fine",
]


def normalize_item_text(text: str) -> str:
    """Repo convention (data_processing/item_records.py): strip only."""
    return str(text).strip()


def item_id_from_text(text: str) -> str:
    """Repo convention: sha256 of the normalized text."""
    return hashlib.sha256(normalize_item_text(text).encode("utf-8")).hexdigest()


@dataclass
class Dataset:
    texts: list[str]  # normalized, CSV row order
    references: dict[str, np.ndarray]  # pov_name -> int labels, aligned to texts

    def __post_init__(self):
        for pov, labels in self.references.items():
            if len(labels) != len(self.texts):
                raise ValueError(
                    f"Reference '{pov}' has {len(labels)} labels for {len(self.texts)} texts"
                )

    @property
    def text_to_index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.texts)}


def load_dataset_csv(path: str | Path, pov_columns: list[str] | None = None) -> Dataset:
    """Load the enterprise-ticket dataset CSV into texts + reference partitions."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row")
        missing = [c for c in [TEXT_COLUMN, *(pov_columns or POV_COLUMNS)] if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing expected columns {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty CSV")

    cols = pov_columns or POV_COLUMNS
    texts = []
    for i, row in enumerate(rows):
        t = normalize_item_text(row[TEXT_COLUMN])
        if not t:
            raise ValueError(f"{path}: blank {TEXT_COLUMN} at data row {i}")
        texts.append(t)
    if len(set(texts)) != len(texts):
        raise ValueError(f"{path}: duplicate ticket_text values ({len(texts) - len(set(texts))})")

    references = {}
    for col in cols:
        values = [row[col] for row in rows]
        blank = sum(1 for v in values if v is None or not str(v).strip())
        if blank:
            raise ValueError(f"{path}: column '{col}' has {blank} blank values")
        distinct = sorted(set(values))
        code = {v: k for k, v in enumerate(distinct)}
        references[col] = np.array([code[v] for v in values])
    return Dataset(texts=texts, references=references)


def load_benchmark_spec_references(path: str | Path) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Benchmark spec JSON -> (items, {reference_id: {item_text: cluster_id}})."""
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    for key in ("items", "references"):
        if key not in spec:
            raise ValueError(f"{path}: missing top-level key '{key}'")
    items = [normalize_item_text(t) for t in spec["items"]]
    refs = {}
    for ref in spec["references"]:
        rid = ref["reference_id"]
        assignment = {normalize_item_text(k): int(v) for k, v in ref["assignment"].items()}
        if set(assignment) != set(items):
            raise ValueError(f"{path}: reference '{rid}' item set differs from spec items")
        refs[rid] = assignment
    return items, refs


# ---------------------------------------------------------------------------
# Batch-run artifacts (Experiment B)
# ---------------------------------------------------------------------------

_METADATA_POV_KEYS = (
    "topic_branch_topic_id",
    "benchmark_theme_id",
    "topic_conditioned_theme_id",
)


@dataclass
class RunPartitions:
    # run_key ("config_slug/run_id") -> {item_text: cluster_id}
    assignments: dict[str, dict[str, int]]
    # run_key -> pov name (only runs whose POV could be resolved)
    povs: dict[str, str]
    skipped_no_clustering: list[str] = field(default_factory=list)
    skipped_no_pov: list[str] = field(default_factory=list)

    def resolved(self) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
        """Only runs with both a final partition and a resolved POV."""
        keep = {k: v for k, v in self.assignments.items() if k in self.povs}
        return keep, {k: self.povs[k] for k in keep}


def _parse_clustering_json(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "clusters" not in payload or not isinstance(payload["clusters"], list):
        raise ValueError(f"{path}: expected top-level 'clusters' list")
    assignment: dict[str, int] = {}
    for idx, cluster in enumerate(payload["clusters"]):
        members = cluster.get("members")
        if members is None:
            raise ValueError(f"{path}: cluster {idx} has no 'members'")
        for m in members:
            t = normalize_item_text(m)
            if t in assignment:
                # item in two clusters: keep first, note loudly
                print(f"WARNING: {path}: item assigned to multiple clusters, keeping first")
                continue
            assignment[t] = idx
    if not assignment:
        raise ValueError(f"{path}: empty clustering")
    return assignment


def load_batch_run_partitions(batch_dir: str | Path) -> RunPartitions:
    """Walk a batch-run output dir and collect final partitions + POVs per run."""
    batch_dir = Path(batch_dir)
    manifest = batch_dir / "results.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found — is this a batch-run output dir?")

    assignments: dict[str, dict[str, int]] = {}
    povs: dict[str, str] = {}
    no_clustering: list[str] = []
    no_pov: list[str] = []

    with manifest.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"{manifest}: empty manifest")

    for rec in records:
        config_slug = rec["config_slug"]
        metadata = (rec.get("config") or {}).get("metadata") or {}
        metadata_pov = next(
            (metadata[k] for k in _METADATA_POV_KEYS if metadata.get(k)), None
        )
        for run_id in rec["run_ids"]:
            run_key = f"{config_slug}/{run_id}"
            run_dir = batch_dir / config_slug / run_id
            clustering_path = run_dir / "clustering.json"
            if not clustering_path.exists():
                no_clustering.append(run_key)
                continue
            assignments[run_key] = _parse_clustering_json(clustering_path)

            pov = metadata_pov
            selected_topic = run_dir / "selected_topic.json"
            if selected_topic.exists():
                topic = json.loads(selected_topic.read_text(encoding="utf-8"))
                pov = topic.get("topic_id") or topic.get("label") or pov
            if pov:
                povs[run_key] = str(pov)
            else:
                no_pov.append(run_key)

    if no_clustering:
        print(f"NOTE: {len(no_clustering)} run(s) have no saved clustering.json")
    if no_pov:
        print(f"NOTE: {len(no_pov)} run(s) have a partition but no resolvable POV")
    return RunPartitions(
        assignments=assignments,
        povs=povs,
        skipped_no_clustering=no_clustering,
        skipped_no_pov=no_pov,
    )


def load_run_partitions_csv(path: str | Path) -> RunPartitions:
    """Long-format export: run_id, pov_name, item_id, cluster_id.

    item_id may be either the sha256 item id or the raw item text; it is used
    only as an opaque join key across runs, so both work.
    """
    path = Path(path)
    assignments: dict[str, dict[str, int]] = {}
    povs: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"run_id", "pov_name", "item_id", "cluster_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {sorted(required)}, got {reader.fieldnames}")
        for i, row in enumerate(reader):
            run_id = row["run_id"].strip()
            if not run_id:
                raise ValueError(f"{path}: blank run_id at row {i}")
            assignments.setdefault(run_id, {})[row["item_id"].strip()] = int(row["cluster_id"])
            pov = row["pov_name"].strip()
            if pov:
                if povs.get(run_id, pov) != pov:
                    raise ValueError(f"{path}: run {run_id} has conflicting pov_name values")
                povs[run_id] = pov
    if not assignments:
        raise ValueError(f"{path}: no rows")
    no_pov = [r for r in assignments if r not in povs]
    if no_pov:
        print(f"NOTE: {len(no_pov)} run(s) in CSV have no pov_name")
    return RunPartitions(assignments=assignments, povs=povs, skipped_no_pov=no_pov)
