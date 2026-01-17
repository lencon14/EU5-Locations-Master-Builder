#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read-only analyzer for artifacts/diagnostic_lake_adjacency.log

Supports:
- Single-line summaries with timestamp/level prefix:
  - lake_adjacency(input): edges=... rgb_map=... lake_ids=... sea_ids=... lake_rgbs=... sea_rgbs=...
  - lake_adjacency(image): N locations adjacent to lakes
- JSON lines and KV/section styles (best-effort)
- compare subcommand (A vs B)

Safety:
- Reads logs only, never touches pipeline outputs
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _try_json_loads(s: str) -> Optional[Any]:
    s = s.strip()
    if not s or not (s.startswith("{") or s.startswith("[")):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _try_literal_eval(s: str) -> Optional[Any]:
    s = s.strip()
    if not s:
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _coerce_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float) and math.isfinite(x):
        return int(x)
    if isinstance(x, str):
        t = x.strip()
        if re.fullmatch(r"[-+]?\d+", t):
            try:
                return int(t)
            except Exception:
                return None
    return None


def _safe_len(x: Any) -> Optional[int]:
    try:
        return len(x)
    except Exception:
        return None


@dataclass
class LakeAdjRecord:
    record_type: str = "unknown"  # input_summary | image_summary | structured | unknown
    location_id: Optional[str] = None
    location_name: Optional[str] = None
    input_payload: Dict[str, Any] = field(default_factory=dict)
    image_payload: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def adjacent(self) -> Optional[int]:
        for k in ("adjacent", "adjacent_count", "adjacent_locations", "lake_adjacent"):
            v = _coerce_int(self.image_payload.get(k))
            if v is not None:
                return v
        return None

    def _count_value(self, v: Any) -> Optional[int]:
        # For summary logs, values are already ints.
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        # For structured payloads, lengths matter.
        if isinstance(v, (list, tuple, set, dict)):
            return _safe_len(v)
        return None

    def counts(self) -> Dict[str, Optional[int]]:
        inp = self.input_payload
        return {
            "edges": self._count_value(inp.get("edges")),
            "rgb_map": self._count_value(inp.get("rgb_map")),
            "lake_ids": self._count_value(inp.get("lake_ids")),
            "sea_ids": self._count_value(inp.get("sea_ids")),
            "lake_rgbs": self._count_value(inp.get("lake_rgbs")),
            "sea_rgbs": self._count_value(inp.get("sea_rgbs")),
            "adjacent": self.adjacent(),
        }


# Strip common prefixes like:
# 2026-01-16 18:40:13,320 INFO ...
TS_LEVEL_PREFIX_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+(?:INFO|DEBUG|WARN|WARNING|ERROR)\s+",
    re.IGNORECASE,
)

INPUT_SUMMARY_RE = re.compile(
    r"^\s*lake_adjacency\(input\)\s*:\s*"
    r"edges=(\d+)\s+rgb_map=(\d+)\s+lake_ids=(\d+)\s+sea_ids=(\d+)\s+lake_rgbs=(\d+)\s+sea_rgbs=(\d+)\s*$",
    re.IGNORECASE,
)
IMAGE_SUMMARY_RE = re.compile(
    r"^\s*lake_adjacency\(image\)\s*:\s*(\d+)\s+locations\s+adjacent\s+to\s+lakes\s*$",
    re.IGNORECASE,
)

SECTION_RE = re.compile(r"^\s*(lake_adjacency\((?:input|image)\))\s*:\s*$", re.IGNORECASE)
KV_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*(=|:)\s*(.*)\s*$")


def _strip_prefix(s: str) -> str:
    return TS_LEVEL_PREFIX_RE.sub("", s, count=1).strip()


def parse_log_best_effort(path: str) -> Tuple[List[LakeAdjRecord], Dict[str, int]]:
    recs: List[LakeAdjRecord] = []
    stats = {"lines_total": 0, "json": 0, "kv": 0, "ignored": 0, "records": 0}

    current: Optional[LakeAdjRecord] = None
    current_section: Optional[str] = None

    def flush() -> None:
        nonlocal current, current_section
        if current and (current.input_payload or current.image_payload or current.raw):
            recs.append(current)
        current = None
        current_section = None

    def ensure() -> LakeAdjRecord:
        nonlocal current
        if current is None:
            current = LakeAdjRecord(record_type="structured")
        return current

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stats["lines_total"] += 1
            s0 = line.rstrip("\n")
            s = s0.strip()
            if not s:
                flush()
                continue

            stripped = _strip_prefix(s)

            mi = INPUT_SUMMARY_RE.match(stripped)
            if mi:
                r = LakeAdjRecord(record_type="input_summary")
                r.input_payload.update(
                    {
                        "edges": int(mi.group(1)),
                        "rgb_map": int(mi.group(2)),
                        "lake_ids": int(mi.group(3)),
                        "sea_ids": int(mi.group(4)),
                        "lake_rgbs": int(mi.group(5)),
                        "sea_rgbs": int(mi.group(6)),
                    }
                )
                recs.append(r)
                continue

            mg = IMAGE_SUMMARY_RE.match(stripped)
            if mg:
                r = LakeAdjRecord(record_type="image_summary")
                r.image_payload["adjacent"] = int(mg.group(1))
                recs.append(r)
                continue

            j = _try_json_loads(stripped)
            if j is not None:
                stats["json"] += 1
                d = j if isinstance(j, dict) else {"_json": j}
                cur = ensure()
                # best-effort: classify by keys
                if any(k in d for k in ("edges", "rgb_map", "lake_ids", "sea_ids", "lake_rgbs", "sea_rgbs")):
                    cur.input_payload.update(d)
                elif any(k in d for k in ("adjacent", "adjacent_count", "adjacent_locations")):
                    cur.image_payload.update(d)
                else:
                    cur.raw.setdefault("json_misc", []).append(d)
                continue

            msec = SECTION_RE.match(stripped)
            if msec:
                current_section = msec.group(1).lower()
                ensure()
                continue

            mkv = KV_RE.match(stripped)
            if mkv:
                stats["kv"] += 1
                key = mkv.group(1)
                val_raw = mkv.group(3).strip()
                val = _try_json_loads(val_raw)
                if val is None:
                    val = _try_literal_eval(val_raw)
                if val is None:
                    val = val_raw

                cur = ensure()
                sec = current_section
                if sec is None:
                    if key in ("adjacent", "adjacent_count", "adjacent_locations", "lake_adjacent"):
                        sec = "lake_adjacency(image)"
                    elif key in ("edges", "rgb_map", "lake_ids", "sea_ids", "lake_rgbs", "sea_rgbs"):
                        sec = "lake_adjacency(input)"

                if sec == "lake_adjacency(input)":
                    cur.input_payload[key] = val
                elif sec == "lake_adjacency(image)":
                    cur.image_payload[key] = val
                else:
                    cur.raw[key] = val
                continue

            stats["ignored"] += 1
            cur = ensure()
            cur.raw.setdefault("ignored_lines", []).append(s0)

    flush()
    stats["records"] = len(recs)
    return recs, stats


def _collect(recs: List[LakeAdjRecord], field: str) -> List[int]:
    out: List[int] = []
    for r in recs:
        v = r.counts().get(field)
        if isinstance(v, int):
            out.append(v)
    return out


def _pct(sorted_vals: List[int], q: float) -> int:
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = int(round((n - 1) * q))
    return sorted_vals[max(0, min(n - 1, idx))]


def _describe(vals: List[int]) -> Dict[str, Any]:
    if not vals:
        return {"count": 0}
    s = sorted(vals)
    d: Dict[str, Any] = {
        "count": len(s),
        "min": s[0],
        "p50": _pct(s, 0.50),
        "p90": _pct(s, 0.90),
        "p95": _pct(s, 0.95),
        "max": s[-1],
    }
    if len(s) >= 2:
        d["mean"] = float(statistics.mean(s))
        d["stdev"] = float(statistics.pstdev(s))
    return d


def _topk(recs: List[LakeAdjRecord], field: str, k: int) -> List[Dict[str, Any]]:
    rows: List[Tuple[int, int, LakeAdjRecord]] = []
    for i, r in enumerate(recs):
        v = r.counts().get(field)
        if isinstance(v, int):
            rows.append((v, i, r))
    rows.sort(key=lambda t: (-t[0], t[1]))
    out: List[Dict[str, Any]] = []
    for rank, (v, idx, r) in enumerate(rows[:k], start=1):
        out.append(
            {
                "rank": rank,
                "value": v,
                "record_index": idx,
                "record_type": r.record_type,
                "location_id": r.location_id,
                "location_name": r.location_name,
            }
        )
    return out


def _anomalies(recs: List[LakeAdjRecord]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(recs):
        c = r.counts()
        reasons: List[str] = []

        if r.record_type == "image_summary":
            if c["adjacent"] is None:
                reasons.append("missing adjacent in image_summary")
        elif r.record_type == "input_summary":
            for k in ("edges", "rgb_map", "lake_ids", "sea_ids", "lake_rgbs", "sea_rgbs"):
                if c[k] is None:
                    reasons.append(f"missing {k} in input_summary")
        # structured/unknown: be conservative (do not flag missing unless both sides absent)
        if reasons:
            out.append(
                {
                    "record_index": i,
                    "record_type": r.record_type,
                    "reasons": reasons,
                    "counts": c,
                }
            )
    return out


def build_report(recs: List[LakeAdjRecord], stats: Dict[str, int], top_n: int) -> Dict[str, Any]:
    fields = ["adjacent", "edges", "rgb_map", "lake_ids", "sea_ids", "lake_rgbs", "sea_rgbs"]
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "records": len(recs),
            "parse": stats,
        },
        "distributions": {f: _describe(_collect(recs, f)) for f in fields},
        "top": {f: _topk(recs, f, top_n) for f in ("adjacent", "edges")},
        "anomalies": _anomalies(recs),
    }


def render(report: Dict[str, Any]) -> str:
    m = report["meta"]
    p = m["parse"]
    lines: List[str] = []
    lines.append("Lake Adjacency Diagnostic Log Summary")
    lines.append("=" * 44)
    lines.append(f"Generated: {m['generated_at']}")
    lines.append(f"Records:   {m['records']}")
    lines.append(f"Parse:     lines={p['lines_total']} json={p['json']} kv={p['kv']} ignored={p['ignored']} records={p['records']}")
    lines.append("")
    lines.append("Distributions")
    lines.append("-" * 44)
    for k, d in report["distributions"].items():
        if d.get("count", 0) == 0:
            lines.append(f"{k:10s}: (no data)")
        else:
            mean = d.get("mean")
            stdev = d.get("stdev")
            mean_s = f"{mean:.2f}" if isinstance(mean, (int, float)) else "n/a"
            stdev_s = f"{stdev:.2f}" if isinstance(stdev, (int, float)) else "n/a"
            lines.append(
                f"{k:10s}: n={d['count']} min={d['min']} p50={d['p50']} p90={d['p90']} max={d['max']} mean={mean_s} stdev={stdev_s}"
            )
    lines.append("")
    lines.append("Top (by adjacent)")
    lines.append("-" * 44)
    top = report["top"]["adjacent"]
    if not top:
        lines.append("(no data)")
    else:
        for row in top:
            lines.append(f"#{row['rank']:>2} adjacent={row['value']}  type={row['record_type']}  id={row['location_id'] or '-'}  name={row['location_name'] or '-'}")
    lines.append("")
    an = report["anomalies"]
    lines.append(f"Anomalies (candidates): {len(an)}")
    lines.append("-" * 44)
    for a in an[:50]:
        lines.append(f"[#{a['record_index']}] type={a['record_type']} :: " + "; ".join(a["reasons"]))
        lines.append("  counts: " + json.dumps(a["counts"], ensure_ascii=False))
    if len(an) > 50:
        lines.append(f"... ({len(an)-50} more)")
    lines.append("")
    return "\n".join(lines)


def compare_reports(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"meta": {"a_records": a["meta"]["records"], "b_records": b["meta"]["records"]}, "delta": {}}
    for k in a["distributions"].keys():
        da = a["distributions"][k]
        db = b["distributions"][k]
        out["delta"][k] = {
            "count": {"a": da.get("count"), "b": db.get("count")},
            "p50": {"a": da.get("p50"), "b": db.get("p50"), "diff": (db.get("p50") - da.get("p50")) if isinstance(da.get("p50"), int) and isinstance(db.get("p50"), int) else None},
            "max": {"a": da.get("max"), "b": db.get("max"), "diff": (db.get("max") - da.get("max")) if isinstance(da.get("max"), int) and isinstance(db.get("max"), int) else None},
        }
    out["anomalies"] = {"a": len(a["anomalies"]), "b": len(b["anomalies"]), "diff": len(b["anomalies"]) - len(a["anomalies"])}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze diagnostic lake adjacency log (read-only).")
    ap.add_argument("--log", default=os.path.join("artifacts", "diagnostic_lake_adjacency.log"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--format", choices=["text", "json"], default="text")
    sub = ap.add_subparsers(dest="cmd")

    cp = sub.add_parser("compare")
    cp.add_argument("--a", required=True)
    cp.add_argument("--b", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "compare":
        ra, sa = parse_log_best_effort(args.a)
        rb, sb = parse_log_best_effort(args.b)
        rep_a = build_report(ra, sa, args.top)
        rep_b = build_report(rb, sb, args.top)
        delta = compare_reports(rep_a, rep_b)
        if args.format == "json":
            print(json.dumps(delta, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(delta, ensure_ascii=False, indent=2))
        return 0

    recs, stats = parse_log_best_effort(args.log)
    rep = build_report(recs, stats, args.top)
    if args.format == "json":
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print(render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
