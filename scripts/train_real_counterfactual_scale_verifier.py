import argparse
import ast
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def extract_numbers(x):
    s = str(x).lower().replace(",", "").replace("−", "-")
    nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", s)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            pass
    return out


def one_number(x):
    nums = extract_numbers(x)
    if len(nums) != 1:
        return None
    return nums[0]


def fmt_num(x):
    if abs(x - round(x)) < 1e-8:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def first_answer(ans):
    s = str(ans)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and obj:
            return str(obj[0])
    except Exception:
        pass
    return s


def make_scaled_wrong(v):
    # skip years, because 1990 vs 19.90 is not a useful scale-pair
    if 1500 <= abs(v) <= 2200:
        return None

    if abs(v) >= 2:
        return v / 100.0
    elif 0 < abs(v) <= 1.5:
        return v * 100.0
    else:
        return None


def is_factor100_conflict(base_pred, comp_pred):
    b = one_number(base_pred)
    c = one_number(comp_pred)
    if b is None or c is None or abs(c) < 1e-12:
        return False
    ratio = abs(b / c)
    return 80 <= ratio <= 125 and abs(c) < abs(b)


def load_margin0(path):
    df = pd.read_csv(path)
    if "margin" in df.columns:
        df["_m"] = pd.to_numeric(df["margin"], errors="coerce")
        df = df[df["_m"].abs() < 1e-9].copy()
    return df.drop_duplicates("idx").copy()


def verifier_prompt(question, cand_a, cand_b):
    return (
        "You are a chart scale verifier.\n"
        "Look carefully at the chart image, especially axis tick labels, value labels, legend, and units.\n"
        "Two candidate answers differ mainly by numerical scale.\n"
        "Choose the candidate that matches the original chart scale.\n"
        "If the chart shows 40, choose 40 rather than 0.40.\n"
        "If the chart shows 0.40, choose 0.40 rather than 40.\n"
        "Do not blindly convert percentages to decimals or decimals to percentages.\n\n"
        f"Question: {question}\n"
        f"Candidate A: {cand_a}\n"
        f"Candidate B: {cand_b}\n\n"
        "Correct choice:"
    )


def get_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def completion_logp_and_nll(model, processor, image, prompt, answer):
    prompt_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    full_messages = prompt_messages + [{
        "role": "assistant",
        "content": [{"type": "text", "text": answer}],
    }]

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(full_messages)
    device = get_device(model)

    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    labels = inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    out = model(**inputs)
    logits = out.logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)

    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    denom = mask.sum().clamp(min=1)
    mean_logp = (tok_logp * mask).sum() / denom
    return mean_logp, -mean_logp


@torch.no_grad()
def score_choice(model, processor, image, prompt):
    logp_a, _ = completion_logp_and_nll(model, processor, image, prompt, "A")
    logp_b, _ = completion_logp_and_nll(model, processor, image, prompt, "B")
    return float(logp_a), float(logp_b), float(logp_b - logp_a)


def evaluate_router(raw_df, deltas, tau):
    rows = []

    for _, r in raw_df.iterrows():
        idx = int(r["idx"])
        selected = "base"
        score = int(r["base_score"])
        selected_pred = r["base_pred"]
        delta = deltas.get(idx, None)

        if idx in deltas and deltas[idx] > tau:
            selected = "comp"
            score = int(r["comp_score"])
            selected_pred = r["comp_pred"]

        rows.append({
            **r.to_dict(),
            "selected_from": selected,
            "selected_pred": selected_pred,
            "router_delta": delta,
            "router_score": score,
        })

    out = pd.DataFrame(rows)
    wins = out[(out["router_score"] == 1) & (out["base_score"] == 0)]
    losses = out[(out["router_score"] == 0) & (out["base_score"] == 1)]

    return {
        "df": out,
        "score": float(out["router_score"].mean()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "net": int(len(wins) - len(losses)),
        "selected": int((out["selected_from"] == "comp").sum()),
    }


def build_counterfactual_rows(train_raw):
    rows = []

    for _, r in train_raw.iterrows():
        gold = first_answer(r["answers"])
        v = one_number(gold)
        if v is None:
            continue

        wrong = make_scaled_wrong(v)
        if wrong is None:
            continue

        correct = fmt_num(v)
        wrong = fmt_num(wrong)

        if correct == wrong:
            continue

        # order 1: correct is A
        rows.append({
            "idx": int(r["idx"]),
            "question": r["question"],
            "candidate_a": correct,
            "candidate_b": wrong,
            "label": "A",
            "source": "counterfactual_A",
        })

        # order 2: correct is B
        rows.append({
            "idx": int(r["idx"]),
            "question": r["question"],
            "candidate_a": wrong,
            "candidate_b": correct,
            "label": "B",
            "source": "counterfactual_B",
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--base-adapter", required=True)
    parser.add_argument("--train-logprob", required=True)
    parser.add_argument("--val-logprob", required=True)
    parser.add_argument("--output-adapter", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-max-samples", type=int, default=1500)
    parser.add_argument("--val-max-samples", type=int, default=500)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--real-conflict-prob", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--loss-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_raw = load_margin0(args.train_logprob)
    val_raw = load_margin0(args.val_logprob)

    cf_rows = build_counterfactual_rows(train_raw)

    train_conf = train_raw[
        train_raw.apply(lambda r: is_factor100_conflict(r["base_pred"], r["comp_pred"]), axis=1)
    ].copy()

    train_sup = train_conf[train_conf["base_score"] != train_conf["comp_score"]].copy()
    train_sup["label"] = train_sup.apply(
        lambda r: "B" if int(r["comp_score"]) > int(r["base_score"]) else "A",
        axis=1,
    )

    real_rows = []
    for _, r in train_sup.iterrows():
        real_rows.append({
            "idx": int(r["idx"]),
            "question": r["question"],
            "candidate_a": r["base_pred"],
            "candidate_b": r["comp_pred"],
            "label": r["label"],
            "source": "real_conflict",
        })

    print("counterfactual rows:", len(cf_rows))
    print("real conflict rows:", len(real_rows))
    print("real choose B:", sum(1 for r in real_rows if r["label"] == "B"))
    print("real choose A:", sum(1 for r in real_rows if r["label"] == "A"))

    print("Loading datasets...")
    train_ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{args.train_split}[:{args.train_max_samples}]")
    val_ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{args.val_split}[:{args.val_max_samples}]")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False

    print("Loading trainable adapter:", args.base_adapter)
    model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    print("Training real-image counterfactual scale verifier...")

    for step in range(args.steps):
        use_real = real_rows and random.random() < args.real_conflict_prob

        if use_real:
            row = random.choice(real_rows)
        else:
            row = random.choice(cf_rows)

        idx = int(row["idx"])
        image = train_ds[idx]["image"]
        prompt = verifier_prompt(row["question"], row["candidate_a"], row["candidate_b"])
        label = row["label"]

        _, nll = completion_logp_and_nll(model, processor, image, prompt, label)

        opt.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 50 == 0 or step == args.steps - 1:
            print(f"step={step:04d} src={row['source']} label={label} loss={nll.item():.4f}")

    out_adapter = Path(args.output_adapter)
    out_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_adapter), safe_serialization=True)
    print("Saved verifier adapter:", out_adapter)

    model.eval()

    def collect_deltas(raw_df, ds, split_name):
        deltas = {}
        conflict_df = raw_df[
            raw_df.apply(lambda r: is_factor100_conflict(r["base_pred"], r["comp_pred"]), axis=1)
        ].copy()

        print(f"\nScoring {split_name} conflicts:", len(conflict_df))

        for _, r in conflict_df.iterrows():
            idx = int(r["idx"])
            image = ds[idx]["image"]
            prompt = verifier_prompt(r["question"], r["base_pred"], r["comp_pred"])
            logp_a, logp_b, delta = score_choice(model, processor, image, prompt)
            deltas[idx] = delta
            print(
                f"[{split_name} {idx}] A={logp_a:.4f} B={logp_b:.4f} "
                f"delta={delta:.4f} base={r['base_pred']} comp={r['comp_pred']}"
            )

        return deltas

    train_deltas = collect_deltas(train_raw, train_ds, "train")
    val_deltas = collect_deltas(val_raw, val_ds, "val")

    candidates = [x / 100 for x in range(-300, 301, 2)]

    best = None
    for tau in candidates:
        res = evaluate_router(train_raw, train_deltas, tau)
        obj = res["score"] - args.loss_penalty * (res["losses"] / max(len(train_raw), 1))
        item = (obj, tau, res)
        if best is None or item[0] > best[0]:
            best = item

    _, best_tau, best_train = best
    val_res = evaluate_router(val_raw, val_deltas, best_tau)

    print("\nChosen tau:", best_tau)
    print("Train router:", round(best_train["score"], 4), "wins", best_train["wins"], "losses", best_train["losses"], "selected", best_train["selected"])

    print("\nVAL SUMMARY")
    print("val base:", round(float(val_raw["base_score"].mean()), 4))
    print("val comp:", round(float(val_raw["comp_score"].mean()), 4))
    print("val oracle:", round(float(val_raw[["base_score", "comp_score"]].max(axis=1).mean()), 4))
    print("val router:", round(val_res["score"], 4))
    print("val wins:", val_res["wins"])
    print("val losses:", val_res["losses"])
    print("val net:", val_res["net"])
    print("val selected:", val_res["selected"])

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    val_res["df"].to_csv(out_csv, index=False)

    print("\nSelected comp cases:")
    sel = val_res["df"][val_res["df"]["selected_from"] == "comp"]
    for _, r in sel.iterrows():
        print("\n" + "=" * 80)
        print("idx:", r["idx"])
        print("Q:", r["question"])
        print("Ans:", r["answers"])
        print("base:", r["base_pred"], "score", r["base_score"])
        print("comp:", r["comp_pred"], "score", r["comp_score"])
        print("delta:", r["router_delta"])

    summary = {
        "train_base": round(float(train_raw["base_score"].mean()), 4),
        "train_comp": round(float(train_raw["comp_score"].mean()), 4),
        "train_oracle": round(float(train_raw[["base_score", "comp_score"]].max(axis=1).mean()), 4),
        "train_router": round(float(best_train["score"]), 4),
        "train_wins": best_train["wins"],
        "train_losses": best_train["losses"],
        "train_selected": best_train["selected"],
        "val_base": round(float(val_raw["base_score"].mean()), 4),
        "val_comp": round(float(val_raw["comp_score"].mean()), 4),
        "val_oracle": round(float(val_raw[["base_score", "comp_score"]].max(axis=1).mean()), 4),
        "val_router": round(float(val_res["score"]), 4),
        "val_wins": val_res["wins"],
        "val_losses": val_res["losses"],
        "val_net": val_res["net"],
        "val_selected": val_res["selected"],
        "tau": best_tau,
        "counterfactual_rows": len(cf_rows),
        "real_conflict_rows": len(real_rows),
    }

    summary_path = out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSaved:", out_csv)
    print("Saved:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
