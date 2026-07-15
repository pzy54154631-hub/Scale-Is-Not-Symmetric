#!/usr/bin/env python3
"""Generate local supplementary diagnostics for the EMNLP revision.

This script intentionally uses only the Python standard library. It consumes
archived prediction/selector CSVs that are already present in the repository or
expanded under /tmp/mirage_archives_for_emnlp.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCH = Path("/tmp/mirage_archives_for_emnlp")
OUT_DIR = ROOT / "results" / "analysis" / "emnlp_supplement_20260524"
TABLE_DIR = ROOT / "paper" / "tables"


V10_RESULTS = ARCH / "20260521_scaleqa_policy_selector_v10b_tta2_noleak" / "results" / "scaleqa_natural" / "policy_selector_v10b_tta2_noleak" / "scaleqa520_policy_selector_v10b_tta2_noleak_results.csv"
V10_CHOICES = ARCH / "20260521_scaleqa_policy_selector_v10b_tta2_noleak" / "results" / "scaleqa_natural" / "policy_selector_v10b_tta2_noleak" / "scaleqa520_policy_selector_v10b_tta2_noleak_choices.csv"
CANDIDATE_POOL = ARCH / "20260521_scaleqa_routing_reranker_taxonomy_plus_fast" / "results" / "scaleqa_natural" / "reranker_v2" / "scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv"
MANIFEST_473 = ROOT / "results" / "scaleqa_natural" / "image_assisted_v3" / "scaleqa_natural_520_image_assisted_manifest.csv"
EXPANDED_520 = ROOT / "results" / "scaleqa_natural" / "image_assisted_v3" / "scaleqa_natural_expanded_520_codex_image_assisted.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt4(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.4f}"


def pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100.0 * x:.1f}\\%"


def tex_escape(s: object) -> str:
    text = str(s)
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]:
        text = text.replace(a, b)
    return text


def mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else float("nan")


def std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def bootstrap_ci(vals: list[float], seed: int = 0, n_boot: int = 20000) -> tuple[float, float, float]:
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(mean([rng.choice(vals) for _ in vals]))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    p_le_0 = sum(v <= 0 for v in boots) / n_boot
    return lo, hi, p_le_0


def make_selector_stability() -> dict[str, object]:
    rows = read_csv(V10_RESULTS)
    selector = [float(r["selector"]) for r in rows]
    self_check = [float(r["self_check"]) for r in rows]
    best_fixed = [float(r["best_fixed_split"]) for r in rows]
    oracle = [float(r["policy_oracle"]) for r in rows]
    switch = [float(r["switch_rate"]) for r in rows]
    delta = [a - b for a, b in zip(selector, self_check)]
    lo, hi, p_le_0 = bootstrap_ci(delta)

    out_rows = [
        {
            "metric": "No-leak selector",
            "mean": mean(selector),
            "std": std(selector),
            "ci_low": "",
            "ci_high": "",
            "note": "50 held-out splits",
        },
        {
            "metric": "Self-check fixed prompt",
            "mean": mean(self_check),
            "std": std(self_check),
            "ci_low": "",
            "ci_high": "",
            "note": "same splits",
        },
        {
            "metric": "Selector - self-check",
            "mean": mean(delta),
            "std": std(delta),
            "ci_low": lo,
            "ci_high": hi,
            "note": f"bootstrap p<=0={p_le_0:.3f}; wins/ties/losses={sum(d > 0 for d in delta)}/{sum(d == 0 for d in delta)}/{sum(d < 0 for d in delta)}",
        },
        {
            "metric": "Best fixed policy per split",
            "mean": mean(best_fixed),
            "std": std(best_fixed),
            "ci_low": "",
            "ci_high": "",
            "note": "upper bound over fixed policies on test split",
        },
        {
            "metric": "Policy oracle",
            "mean": mean(oracle),
            "std": std(oracle),
            "ci_low": "",
            "ci_high": "",
            "note": "upper bound",
        },
        {
            "metric": "Selector switch rate",
            "mean": mean(switch),
            "std": std(switch),
            "ci_low": "",
            "ci_high": "",
            "note": "fraction not using self-check",
        },
    ]
    fields = ["metric", "mean", "std", "ci_low", "ci_high", "note"]
    write_csv(OUT_DIR / "selector_stability_ci.csv", out_rows, fields)

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Quantity & Mean & Std. & Additional statistic \\",
        r"\midrule",
    ]
    lines.append(f"{tex_escape('No-leak selector')} & {fmt4(mean(selector))} & {fmt4(std(selector))} & 50 held-out splits \\\\")
    lines.append(f"{tex_escape('Self-check prompt')} & {fmt4(mean(self_check))} & {fmt4(std(self_check))} & same splits \\\\")
    lines.append(
        f"{tex_escape('Selector - self-check')} & {fmt4(mean(delta))} & {fmt4(std(delta))} & "
        f"95\\% CI [{fmt4(lo)}, {fmt4(hi)}] \\\\"
    )
    lines.append(f"{tex_escape('Best fixed policy')} & {fmt4(mean(best_fixed))} & {fmt4(std(best_fixed))} & split oracle over fixed policies \\\\")
    lines.append(f"{tex_escape('Policy oracle')} & {fmt4(mean(oracle))} & {fmt4(std(oracle))} & upper bound \\\\")
    lines.append(f"{tex_escape('Switch rate')} & {pct(mean(switch))} & {pct(std(switch))} & not using self-check \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLE_DIR / "selector_stability_ci.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "selector_mean": mean(selector),
        "selfcheck_mean": mean(self_check),
        "delta_mean": mean(delta),
        "delta_ci": [lo, hi],
        "delta_bootstrap_p_le_0": p_le_0,
        "wins_ties_losses": [sum(d > 0 for d in delta), sum(d == 0 for d in delta), sum(d < 0 for d in delta)],
        "policy_oracle_mean": mean(oracle),
        "switch_rate_mean": mean(switch),
    }


def candidate_pool_scores() -> dict[str, dict[str, float]]:
    by_id: dict[str, dict[str, float]] = defaultdict(dict)
    for row in read_csv(CANDIDATE_POOL):
        try:
            score = float(row["score"])
        except (TypeError, ValueError):
            continue
        by_id[row["candidate_id"]][row["adapter"]] = score
    return by_id


def make_group_diagnostics() -> dict[str, object]:
    meta = {r["candidate_id"]: r for r in read_csv(MANIFEST_473)}
    choices = read_csv(V10_CHOICES)
    by_id_scores = candidate_pool_scores()

    selector_by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    switch_by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in choices:
        m = meta.get(row["candidate_id"])
        if not m:
            continue
        for key in ["scale_directions", "scale_error_subtype", "source_split"]:
            group = m.get(key, "")
            selector_by_group[(key, group)].append(float(row["score"]))
            switch_by_group[(key, group)].append(0.0 if row["chosen_policy"] == "tta:self_check" else 1.0)

    fixed_by_group: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ids_by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    fixed_policies = ["main:answer_tuned", "main:large_seed5", "tta:self_check", "tta:axis", "tta:unit_guard"]
    for cid, scores in by_id_scores.items():
        m = meta.get(cid)
        if not m:
            continue
        for key in ["scale_directions", "scale_error_subtype", "source_split"]:
            group = m.get(key, "")
            ids_by_group[(key, group)].add(cid)
            for pol in fixed_policies:
                if pol in scores:
                    fixed_by_group[(key, group)][pol].append(scores[pol])
            if scores:
                fixed_by_group[(key, group)]["oracle17"].append(max(scores.values()))

    selected = []
    # Keep the table compact: all directions, major source splits, and major subtypes.
    priority = []
    for group in ["prediction_too_large", "prediction_too_small"]:
        priority.append(("scale_directions", group, "Direction: " + group.replace("_", " ")))
    for group in ["test", "val", "train", "train_shard1"]:
        priority.append(("source_split", group, "Source: " + group.replace("_", " ")))
    for group in ["decimal_to_percent", "percent_to_decimal", "value_scaled_down", "value_scaled_up"]:
        priority.append(("scale_error_subtype", group, "Subtype: " + group.replace("_", " ")))

    for key, group, label in priority:
        k = (key, group)
        d = fixed_by_group.get(k, {})
        selected.append(
            {
                "group": label,
                "n_items": len(ids_by_group.get(k, set())),
                "selector_heldout": mean(selector_by_group.get(k, [])),
                "self_check": mean(d.get("tta:self_check", [])),
                "best_global_probe": max([mean(d.get(p, [])) for p in fixed_policies if d.get(p, [])] or [float("nan")]),
                "oracle17": mean(d.get("oracle17", [])),
                "switch_rate": mean(switch_by_group.get(k, [])),
            }
        )

    fields = [
        "group",
        "n_items",
        "selector_heldout",
        "answer_tuned",
        "large_seed5",
        "self_check",
        "best_global_probe",
        "oracle17",
        "switch_rate",
    ]
    for row in selected:
        k_label = row["group"].split(": ", 1)
        # Recover fixed-policy means for compact table columns.
        if len(k_label) == 2:
            if k_label[0] == "Direction":
                k = ("scale_directions", k_label[1].replace(" ", "_"))
            elif k_label[0] == "Source":
                k = ("source_split", k_label[1].replace(" ", "_"))
            else:
                k = ("scale_error_subtype", k_label[1].replace(" ", "_"))
            d = fixed_by_group.get(k, {})
            row["answer_tuned"] = mean(d.get("main:answer_tuned", []))
            row["large_seed5"] = mean(d.get("main:large_seed5", []))
        else:
            row["answer_tuned"] = float("nan")
            row["large_seed5"] = float("nan")
    write_csv(OUT_DIR / "selector_group_diagnostics.csv", selected, fields)

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Group & N & Selector & Ans. & Large & Self-check & Best fixed \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{tex_escape(row['group'])} & {row['n_items']} & {fmt4(row['selector_heldout'])} & "
            f"{fmt4(row['answer_tuned'])} & {fmt4(row['large_seed5'])} & "
            f"{fmt4(row['self_check'])} & {fmt4(row['best_global_probe'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLE_DIR / "selector_group_diagnostics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"groups": selected}


def make_source_audit() -> dict[str, object]:
    manifest = read_csv(MANIFEST_473)
    expanded = [r for r in read_csv(EXPANDED_520) if r.get("review_label") == "yes"]

    source_counter = Counter(r.get("source_split", "") or "missing" for r in manifest)
    rows = []
    for source, count in source_counter.most_common():
        rows.append(
            {
                "set": "Image-evaluable 473",
                "source": source,
                "count": count,
                "share": count / len(manifest),
            }
        )

    flag_rows = []
    for flag, label in [
        ("contains_chartqa_train_occurrence", "Contains ChartQA train occurrence"),
        ("contains_chartqa_val_occurrence", "Contains ChartQA validation occurrence"),
        ("contains_chartqa_test_occurrence", "Contains ChartQA test occurrence"),
    ]:
        true_count = sum(1 for r in expanded if str(r.get(flag, "")).lower() == "true")
        known_count = sum(1 for r in expanded if str(r.get(flag, "")).lower() in {"true", "false"})
        flag_rows.append(
            {
                "set": "Final 520",
                "source": label,
                "count": true_count,
                "share": true_count / len(expanded),
                "known_count": known_count,
            }
        )

    fields = ["set", "source", "count", "share"]
    write_csv(OUT_DIR / "scaleqa_source_split_audit.csv", rows, fields)
    write_csv(OUT_DIR / "scaleqa_occurrence_flag_audit.csv", flag_rows, ["set", "source", "count", "share", "known_count"])

    compact_rows = rows[:6] + flag_rows
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Audit view & Source / flag & Count & Share \\",
        r"\midrule",
    ]
    for row in compact_rows:
        lines.append(
            f"{tex_escape(row['set'])} & {tex_escape(row['source'])} & {row['count']} & {pct(row['share'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLE_DIR / "scaleqa_source_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"source_counts": rows, "occurrence_flags": flag_rows}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "selector_stability": make_selector_stability(),
        "group_diagnostics": make_group_diagnostics(),
        "source_audit": make_source_audit(),
    }
    (OUT_DIR / "emnlp_supplement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["selector_stability"], indent=2))
    print(f"Wrote diagnostics to {OUT_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
