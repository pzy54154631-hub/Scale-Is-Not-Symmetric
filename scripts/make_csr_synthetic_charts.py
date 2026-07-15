import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt


def fmt_num(x):
    if abs(x - round(x)) < 1e-8:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def make_target(chart_type, question_type, categories, values, operation, answer, relevant=None, extracted=None):
    return {
        "chart_type": chart_type,
        "question_type": question_type,
        "relevant_entities": relevant or [],
        "extracted_values": extracted or [],
        "operation": operation,
        "computed_answer": fmt_num(answer) if isinstance(answer, (int, float)) else str(answer),
        "final_answer": fmt_num(answer) if isinstance(answer, (int, float)) else str(answer),
    }


def build_one(i, out_img_dir):
    chart_type = random.choice(["bar_chart", "line_chart"])
    n = random.randint(4, 6)
    categories = [chr(ord("A") + j) for j in range(n)]
    values = [random.randint(5, 95) for _ in range(n)]

    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=120)

    if chart_type == "bar_chart":
        ax.bar(categories, values)
    else:
        ax.plot(categories, values, marker="o")

    ax.set_title(random.choice(["Survey Result", "Annual Statistics", "Group Comparison", "Measured Values"]))
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)

    img_path = out_img_dir / f"csr_{i:06d}.png"
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)

    qtype = random.choice(["direct_read", "sum", "difference", "max", "min", "count_above"])

    if qtype == "direct_read":
        idx = random.randrange(n)
        question = f"What is the value of category {categories[idx]}?"
        answer = values[idx]
        target = make_target(
            chart_type, qtype, categories, values,
            operation=f"read value of {categories[idx]}",
            answer=answer,
            relevant=[categories[idx]],
            extracted=[values[idx]],
        )

    elif qtype == "sum":
        a, b = random.sample(range(n), 2)
        question = f"What is the total value of category {categories[a]} and category {categories[b]}?"
        answer = values[a] + values[b]
        target = make_target(
            chart_type, qtype, categories, values,
            operation="sum",
            answer=answer,
            relevant=[categories[a], categories[b]],
            extracted=[values[a], values[b]],
        )

    elif qtype == "difference":
        a, b = random.sample(range(n), 2)
        question = f"What is the difference between category {categories[a]} and category {categories[b]}?"
        answer = abs(values[a] - values[b])
        target = make_target(
            chart_type, qtype, categories, values,
            operation="absolute_difference",
            answer=answer,
            relevant=[categories[a], categories[b]],
            extracted=[values[a], values[b]],
        )

    elif qtype == "max":
        idx = max(range(n), key=lambda k: values[k])
        question = "Which category has the highest value?"
        answer = categories[idx]
        target = make_target(
            chart_type, qtype, categories, values,
            operation="argmax",
            answer=answer,
            relevant=[categories[idx]],
            extracted=[values[idx]],
        )

    elif qtype == "min":
        idx = min(range(n), key=lambda k: values[k])
        question = "Which category has the lowest value?"
        answer = categories[idx]
        target = make_target(
            chart_type, qtype, categories, values,
            operation="argmin",
            answer=answer,
            relevant=[categories[idx]],
            extracted=[values[idx]],
        )

    else:
        threshold = random.choice([20, 30, 40, 50, 60, 70])
        answer = sum(v > threshold for v in values)
        question = f"How many categories have values above {threshold}?"
        target = make_target(
            chart_type, qtype, categories, values,
            operation=f"count values greater than {threshold}",
            answer=answer,
            relevant=[c for c, v in zip(categories, values) if v > threshold],
            extracted=[v for v in values if v > threshold],
        )

    return {
        "image": str(img_path),
        "question": question,
        "answer": str(answer),
        "csr_target": target,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/csr_synth")
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out = Path(args.output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    jsonl = out / "csr_synth.jsonl"

    with jsonl.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            ex = build_one(i, img_dir)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("Saved:", jsonl)


if __name__ == "__main__":
    main()
