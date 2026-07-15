import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt


def fmt_num(x):
    if abs(x - round(x)) < 1e-8:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def build_one(i, out_img_dir):
    chart_type = random.choice(["bar", "line"])

    # percent chart: chart visually shows 68, correct answer should be 68
    # decimal chart: chart visually shows 0.68, correct answer should be 0.68
    scale_type = random.choice(["percent", "decimal"])

    n = random.randint(3, 6)
    categories = [f"C{j+1}" for j in range(n)]
    raw_ints = [random.randint(5, 95) for _ in range(n)]

    if scale_type == "percent":
        values = raw_ints
        ylabel = random.choice(["Percent (%)", "Percentage", "Value (%)", "Share (%)"])
    else:
        values = [x / 100.0 for x in raw_ints]
        ylabel = random.choice(["Rate", "Proportion", "Value", "Share", "Index (0-1)"])

    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=120)

    if chart_type == "bar":
        ax.bar(categories, values)
    else:
        ax.plot(categories, values, marker="o")

    ax.set_title(random.choice(["Survey Results", "Group Comparison", "Measured Values", "Annual Statistics"]))
    ax.set_xlabel("Category")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    # make axis scale visually clear
    if scale_type == "percent":
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, 1.0)

    img_path = out_img_dir / f"scale_synth_{i:06d}.png"
    fig.tight_layout()
    fig.savefig(img_path)
    plt.close(fig)

    qtype = random.choice(["direct", "max", "min", "difference", "average"])

    if qtype == "direct":
        idx = random.randrange(n)
        target = values[idx]
        q = random.choice([
            f"What is the value of {categories[idx]}?",
            f"What is the percentage of {categories[idx]}?",
            f"What percent value is shown for {categories[idx]}?",
            f"What is the rate of {categories[idx]}?",
        ])

    elif qtype == "max":
        target = max(values)
        q = random.choice([
            "What is the highest value in the chart?",
            "What is the largest percentage shown?",
            "What is the maximum rate shown?",
        ])

    elif qtype == "min":
        target = min(values)
        q = random.choice([
            "What is the lowest value in the chart?",
            "What is the smallest percentage shown?",
            "What is the minimum rate shown?",
        ])

    elif qtype == "difference":
        hi = max(values)
        lo = min(values)
        target = hi - lo
        q = random.choice([
            "What is the difference between the largest and smallest values?",
            "What is the difference between the highest and lowest percentages?",
            "What is the difference between the maximum and minimum rates?",
        ])

    else:
        target = sum(values) / len(values)
        q = random.choice([
            "What is the average value in the chart?",
            "What is the mean percentage shown?",
            "What is the average rate shown?",
        ])

    # Candidate A always uses the large-scale form.
    # Candidate B always uses the decimal-scale form.
    if scale_type == "percent":
        cand_a = fmt_num(target)
        cand_b = fmt_num(target / 100.0)
        label = "A"
    else:
        cand_a = fmt_num(target * 100.0)
        cand_b = fmt_num(target)
        label = "B"

    return {
        "idx": i,
        "image_path": str(img_path),
        "chart_type": chart_type,
        "scale_type": scale_type,
        "question_type": qtype,
        "question": q,
        "candidate_a": cand_a,
        "candidate_b": cand_b,
        "label": label,
        "answer": cand_a if label == "A" else cand_b,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/scale_synth")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "scale_synth.csv"

    rows = []
    for i in range(args.n):
        rows.append(build_one(i, img_dir))
        if i % 500 == 0:
            print("generated", i)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", csv_path)
    print("Images:", img_dir)
    print("N:", len(rows))


if __name__ == "__main__":
    main()
