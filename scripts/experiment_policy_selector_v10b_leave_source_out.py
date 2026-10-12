"""Leave-source-out evaluation of the v10b label-free selector.

Reuses the archived v10b feature builder and model zoo unchanged. Instead of
random 60/20/20 splits, each ChartQA source split that contributed ScaleQA
items is held out in turn; the selector (model, mode, margin) is chosen on a
dev partition drawn only from the remaining sources and evaluated once on the
held-out source. Run from the v10b evidence directory.
"""
import copy, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

EVID = Path("artifacts/evidence/expanded/20260521_scaleqa_policy_selector_v10b_tta2_noleak")
POOL = Path("artifacts/evidence/expanded/20260521_scaleqa_routing_reranker_taxonomy_plus_fast/results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv")
OUT = Path("results/scaleqa_natural/leave_source_out_v10b"); OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("v10b", EVID / "scripts/experiment_policy_selector_v10b_tta2_noleak.py")
v10b = importlib.util.module_from_spec(spec)
import os; cwd = os.getcwd(); os.chdir(EVID)
# the archived script resolves its inputs relative to its evidence dir
v10b.POOL = Path("../20260521_scaleqa_routing_reranker_taxonomy_plus_fast/results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv")
spec.loader.exec_module(v10b)
v10b.POOL = Path("../20260521_scaleqa_routing_reranker_taxonomy_plus_fast/results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv")
ids, policies, score, X_base, X_row, y_row, row_cids, row_meta = v10b.build()
os.chdir(cwd)

src = pd.read_csv(POOL).drop_duplicates("candidate_id").set_index("candidate_id").source_split
src = src.str.replace(r"train_shard\d", "train_shard", regex=True)
groups = {"train_shard1": ["train_shard"], "val": ["val"], "test": ["test"], "train": ["train"]}
default = "tta:self_check"; margins = np.linspace(-0.05, 0.30, 15)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
row_models = {
    "logreg": LogisticRegression(max_iter=2000, C=0.7, class_weight="balanced"),
    "et_small": ExtraTreesClassifier(n_estimators=180, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=4),
    "rf_small": RandomForestClassifier(n_estimators=160, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=4),
}
records = []
for gname, members in groups.items():
    test = np.asarray([c for c in ids if src[c] in members]); rest = np.asarray([c for c in ids if src[c] not in members])
    for seed in range(10):
        rng = np.random.default_rng(seed); sh = rest.copy(); rng.shuffle(sh)
        train, dev = sh[: int(0.75 * len(sh))], sh[int(0.75 * len(sh)):]
        m = lambda s: np.isin(row_cids, s)
        cands = []
        dev_scores = {p: v10b.score_policy(score, dev, p) for p in policies}
        fixed_p = max(dev_scores, key=dev_scores.get); cands.append((dev_scores[fixed_p], "fixed", fixed_p, None))
        for name, clf in row_models.items():
            model = copy.deepcopy(clf); model.set_params(random_state=seed); model.fit(X_row[m(train)], y_row[m(train)])
            probs = model.predict_proba(X_row[m(dev)])[:, 1]; dmeta = [x for x, k in zip(row_meta, m(dev)) if k]
            cands.append((v10b.score_choices(score, v10b.choose_stack(probs, dmeta, dev)), name, "stack", None))
            for mg in margins:
                cands.append((v10b.score_choices(score, v10b.choose_patch(probs, dmeta, dev, default, float(mg))), name, "patch", float(mg)))
        _, name, mode, mg = max(cands, key=lambda x: x[0])
        if name == "fixed":
            choices = {c: mode for c in test}
        else:
            model = copy.deepcopy(row_models[name]); model.set_params(random_state=seed); model.fit(X_row[m(rest)], y_row[m(rest)])
            probs = model.predict_proba(X_row[m(test)])[:, 1]; tmeta = [x for x, k in zip(row_meta, m(test)) if k]
            choices = v10b.choose_stack(probs, tmeta, test) if mode == "stack" else v10b.choose_patch(probs, tmeta, test, default, mg)
        records.append(dict(heldout_source=gname, n_test=len(test), n_train=len(rest), seed=seed, selected=f"{name}:{mode}:{mg}",
            selector=v10b.score_choices(score, choices), self_check=v10b.score_policy(score, test, default),
            dev_selected_fixed=v10b.score_policy(score, test, fixed_p), best_fixed_split=float(score.loc[test, policies].mean().max()),
            policy_oracle=float(score.loc[test, policies].max(axis=1).mean()), switch_rate=float(np.mean([choices[c] != default for c in test]))))
df = pd.DataFrame(records); df.to_csv(OUT / "leave_source_out_v10b_results.csv", index=False)
summ = df.groupby("heldout_source").agg(n_test=("n_test", "first"), n_train=("n_train", "first"), selector=("selector", "mean"), selector_std=("selector", "std"),
    self_check=("self_check", "first"), dev_selected_fixed=("dev_selected_fixed", "mean"), best_fixed=("best_fixed_split", "first"), oracle=("policy_oracle", "first"), switch=("switch_rate", "mean"))
summ["delta"] = summ.selector - summ.self_check
summ.to_csv(OUT / "leave_source_out_v10b_summary.csv"); print(summ.round(4).to_string())
