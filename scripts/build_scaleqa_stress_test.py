#!/usr/bin/env python3
"""Build a ChartQA-derived scale stress test.

The output is a self-contained diagnostic set for scale consistency.  Each row
keeps the original ChartQA image and question, the gold answer, and one
deliberately scaled negative answer.  The same source example can produce two
rows: a divide-by-100 distractor and a multiply-by-100 distractor.

The dataset is intended for two complementary evaluations:

1. Free-form QA: run a model on the original question and check whether the
   answer matches the gold value or a scaled distractor.
2. Forced choice: ask the model to choose between the gold value and the scaled
   distractor using the provided candidate fields.

The script uses deterministic filters instead of an LLM judge so that the
benchmark is reproducible and cheap to rebuild.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable


NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?")


SCALE_KEYWORDS = {
    "%",
    "percent",
    "percentage",
    "rate",
    "ratio",
    "average",
    "mean",
    "median",
    "sum",
    "total",
    "difference",
    "value",
    "number",
    "amount",
    "how many",
    "how much",
    "greater",
    "less",
    "highest",
    "lowest",
    "maximum",
    "minimum",
}

YEAR_KEYWORDS = {
    "year",
    "month",
    "date",
    "when",
}


@dataclass
class StressItem:
    stress_id: str
    source_dataset: str
    source_split: str
    source_idx: int
    image_path: str
    question: str
    answer: str
    gold_value: str
    negative_answer: str
    scale_factor: str
    pair_direction: str
    candidate_a: str
    candidate_b: str
    label: str
    forced_choice_prompt: str
    filter_notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceM4/ChartQA")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--source-csv",
        default=None,
        help="Optional ChartQA-style prediction CSV. When set, build from idx/question/answers without loading HF images.",
    )
    parser.add_argument(
        "--source-parquet",
        default=None,
        help="Optional local ChartQA parquet file. This bypasses the Hugging Face dataset loader and can save cached image bytes.",
    )
    parser.add_argument("--output-dir", default="datasets/scaleqa_stress/chartqa_val")
    parser.add_argument("--max-source-examples", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--include-count-questions", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def normalize_answer_cell(x: Any) -> str:
    if isinstance(x, list):
        return str(x[0]) if x else ""
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            if isinstance(obj, list) and obj:
                return str(obj[0])
        except Exception:
            pass
        try:
            obj = ast.literal_eval(x)
            if isinstance(obj, list) and obj:
                return str(obj[0])
        except Exception:
            pass
    return str(x)


def extract_single_number(text: Any) -> Decimal | None:
    s = str(text).replace(",", "").replace("−", "-").strip()
    nums = NUM_RE.findall(s)
    if len(nums) != 1:
        return None
    try:
        return Decimal(nums[0])
    except InvalidOperation:
        return None


def is_probably_year(question: str, value: Decimal) -> bool:
    q = question.lower()
    if any(k in q for k in YEAR_KEYWORDS):
        return True
    if value == value.to_integral_value() and Decimal(1900) <= value <= Decimal(2100):
        return True
    return False


def has_scale_keyword(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in SCALE_KEYWORDS)


def is_count_question(question: str) -> bool:
    q = question.lower()
    return (
        "how many" in q
        or "number of" in q
        or "count" in q
        or "how much times" in q
    )


def decimal_places(x: Decimal) -> int:
    exponent = x.normalize().as_tuple().exponent
    return max(0, -exponent)


def format_decimal(x: Decimal, reference: Decimal) -> str:
    if not x.is_finite():
        raise ValueError(f"non-finite decimal: {x}")

    if x == 0:
        return "0"

    places = max(decimal_places(reference) + 2, 2)
    if abs(x) >= 1:
        places = min(max(decimal_places(reference), 0), 4)
    places = min(places, 6)

    quant = Decimal(1).scaleb(-places)
    y = x.quantize(quant, rounding=ROUND_HALF_UP)
    s = format(y, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def scaled_candidates(value: Decimal) -> Iterable[tuple[str, Decimal]]:
    yield "divide_100", value / Decimal(100)
    yield "multiply_100", value * Decimal(100)


def make_forced_choice_prompt(question: str, a: str, b: str) -> str:
    return (
        f"{question}\n"
        "Choose the answer that matches the scale used by the chart and the question.\n"
        f"A. {a}\n"
        f"B. {b}\n"
        "Answer with only A or B."
    )


def get_image_and_fields(example: dict[str, Any]) -> tuple[Any, str, str]:
    image = example.get("image")
    question = example.get("query") or example.get("question") or ""
    answer = normalize_answer_cell(example.get("label") or example.get("answer") or example.get("answers"))
    return image, str(question), answer


def load_chartqa(dataset_name: str, split: str, max_source_examples: int | None, offline: bool):
    from datasets import load_dataset

    split_expr = split
    if max_source_examples is not None:
        split_expr = f"{split}[:{max_source_examples}]"
    return load_dataset(dataset_name, split=split_expr, download_mode="reuse_dataset_if_exists" if offline else None)


def load_from_source_csv(path: Path, max_source_examples: int | None) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "adapter" in row and row["adapter"] != "base":
                continue
            idx = int(row.get("idx", len(rows)))
            if idx in seen:
                continue
            seen.add(idx)
            rows.append(
                {
                    "idx": idx,
                    "query": row.get("question", ""),
                    "label": row.get("answers", ""),
                    "image": None,
                }
            )
            if max_source_examples is not None and len(rows) >= max_source_examples:
                break
    return rows


def load_from_source_parquet(path: Path, max_source_examples: int | None) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    if max_source_examples is not None:
        df = df.iloc[:max_source_examples]
    records = df.to_dict("records")
    for i, row in enumerate(records):
        row.setdefault("idx", i)
    return records


def save_image_like(image: Any, path: Path) -> bool:
    if image is None:
        return False
    try:
        if hasattr(image, "save"):
            image.save(path)
            return True
        if isinstance(image, dict):
            if image.get("bytes") is not None:
                path.write_bytes(image["bytes"])
                return True
            if image.get("path"):
                from PIL import Image

                Image.open(image["path"]).save(path)
                return True
        if isinstance(image, (bytes, bytearray)):
            path.write_bytes(bytes(image))
            return True
    except Exception:
        return False
    return False


def write_outputs(items: list[StressItem], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "scaleqa_stress_test.jsonl"
    csv_path = output_dir / "scaleqa_stress_test.csv"
    summary_path = output_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    if items:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(items[0]).keys()))
            writer.writeheader()
            for item in items:
                writer.writerow(asdict(item))

    summary = {
        "n_items": len(items),
        "n_source_examples": len({x.source_idx for x in items}),
        "pair_direction_counts": count_by(items, "pair_direction"),
        "scale_factor_counts": count_by(items, "scale_factor"),
        "source_dataset": items[0].source_dataset if items else None,
        "source_split": items[0].source_split if items else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved JSONL: {jsonl_path}")
    print(f"Saved CSV:   {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def count_by(items: list[StressItem], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(getattr(item, field))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    if args.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    if args.source_csv and args.source_parquet:
        raise ValueError("Use only one of --source-csv and --source-parquet.")
    if args.source_csv:
        ds = load_from_source_csv(Path(args.source_csv), args.max_source_examples)
    elif args.source_parquet:
        ds = load_from_source_parquet(Path(args.source_parquet), args.max_source_examples)
    else:
        ds = load_chartqa(args.dataset, args.split, args.max_source_examples, args.offline)
    items: list[StressItem] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for source_pos, example in enumerate(ds):
        source_idx = int(example.get("idx", source_pos)) if isinstance(example, dict) else source_pos
        image, question, answer = get_image_and_fields(example)
        value = extract_single_number(answer)
        if value is None:
            skip("non_single_numeric_answer")
            continue
        if value == 0 or not math.isfinite(float(value)):
            skip("zero_or_nonfinite")
            continue
        if is_probably_year(question, value):
            skip("year_or_date_like")
            continue
        if not has_scale_keyword(question):
            skip("no_scale_keyword")
            continue
        if is_count_question(question) and not args.include_count_questions:
            skip("count_question")
            continue

        image_path = ""
        if args.save_images and image is not None:
            image_path = str(image_dir / f"chartqa_val_{source_idx:05d}.png")
            if not save_image_like(image, Path(image_path)):
                image_path = ""
                skip("image_save_failed")
        elif image is not None:
            image_path = f"hf://{args.dataset}/{args.split}/{source_idx}"
        else:
            image_path = f"hf://{args.dataset}/{args.split}/{source_idx}"

        gold = format_decimal(value, value)
        for factor_name, negative_value in scaled_candidates(value):
            if negative_value == value or negative_value == 0:
                continue
            negative = format_decimal(negative_value, value)
            if negative == gold:
                continue

            pair_direction = (
                "large_positive"
                if abs(value) > abs(negative_value)
                else "small_positive"
            )

            if rng.random() < 0.5:
                candidate_a, candidate_b, label = gold, negative, "A"
            else:
                candidate_a, candidate_b, label = negative, gold, "B"

            stress_id = f"chartqa_val_{source_idx:05d}_{factor_name}"
            items.append(
                StressItem(
                    stress_id=stress_id,
                    source_dataset="ChartQA",
                    source_split=args.split,
                    source_idx=source_idx,
                    image_path=image_path,
                    question=question,
                    answer=answer,
                    gold_value=gold,
                    negative_answer=negative,
                    scale_factor=factor_name,
                    pair_direction=pair_direction,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                    label=label,
                    forced_choice_prompt=make_forced_choice_prompt(question, candidate_a, candidate_b),
                    filter_notes="numeric_single;scale_keyword;not_year;not_count"
                    if not args.include_count_questions
                    else "numeric_single;scale_keyword;not_year",
                )
            )
            if args.max_items and len(items) >= args.max_items:
                break
        if args.max_items and len(items) >= args.max_items:
            break

    write_outputs(items, output_dir)
    (output_dir / "skipped_counts.json").write_text(
        json.dumps(dict(sorted(skipped.items())), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Skipped:", json.dumps(dict(sorted(skipped.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
