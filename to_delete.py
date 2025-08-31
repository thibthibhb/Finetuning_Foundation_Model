#!/usr/bin/env python3
"""
plan_missing_4c_runs.py

Plan the missing paired runs for task granularity (4-class Light=N1+N2).
For each subject that has 5-class runs but no 4-class runs, infer the minimal
hyperparameters to mirror from that subject's 5-class runs and print a ready
command line for finetune_main.py.

Usage:
  python to_delete.py --csv Plot_Clean/data/all_runs_flat.csv
"""

import argparse
import pandas as pd
from collections import Counter

# ---- defaults from finetune_main.py (used to suppress redundant flags) ----
TRAIN_DEFAULTS = dict(
    epochs=100,
    batch_size=64,
    lr=5e-5,
    weight_decay=0.0,
    optimizer="AdamW",
    scheduler="cosine",
    dropout=0.0,
    sample_rate=200,
    two_phase_training=False,
    phase1_epochs=3,
    head_lr=1e-3,
    backbone_lr=1e-5,
    use_focal_loss=False,
    use_class_weights=False,
    use_amp=False,
    preprocess=False,
    data_ORP=0.6,
    head_type="simple",
    icl_mode="none",
    icl_hidden=256,
    icl_layers=2,
    focal_gamma=2.0,
)

def first_present(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def most_common(series):
    vals = [v for v in series.dropna().tolist()]
    if not vals: return None
    return Counter(vals).most_common(1)[0][0]

def uniq_sorted(series, cast=int):
    vals = [cast(v) for v in series.dropna().tolist()]
    return sorted(sorted(set(vals)))

def add_flag(parts, flag, value, default_sentinel=object(), boolean=False):
    """Append CLI flag if (a) boolean True or (b) value != default."""
    if value is None: return
    if boolean:
        if bool(value): parts.append(f"--{flag}")
        return
    if default_sentinel is not object() and value == default_sentinel:
        return
    parts.append(f"--{flag} {value}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--script", default="cbramod/training/finetuning/finetune_main.py")
    ap.add_argument("--downstream_dataset", default="IDUN_EEG")
    ap.add_argument("--datasets_dir", default="data/datasets/final_dataset")
    ap.add_argument("--model_dir", default="./saved_models")
    ap.add_argument("--label_4c", default="4c-v0", choices=["4c-v0","4c-v1"])
    ap.add_argument("--topk_seeds", type=int, default=None,
                    help="Optional: limit to first K seeds (sorted) to mirror")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # --- resolve column names (robust to your logging variants) ---
    subj_col   = first_present(df, ["cfg.subject_id","subject_id"])
    scheme_col = first_present(df, ["label_scheme","config"])
    state_col  = first_present(df, ["state"])

    # hyperparams we want to mirror (many are optional if not logged)
    cols = dict(
        seed = first_present(df, ["seeds","seed","cfg.seed","contract.training.seed"]),
        n_train_subjects = first_present(df, ["contract.training.n_train_subjects","n_train_subjects","T_star","n_train_subj"]),
        # train pool / split reproducibility:
        train_subjects_yaml = first_present(df, ["contract.training.train_subjects_yaml","train_subjects_yaml","loso_yaml"]),
        datasets = first_present(df, ["datasets","contract.datasets","cfg.datasets"]),
        # general training knobs:
        epochs = first_present(df, ["epochs","contract.training.epochs"]),
        batch_size = first_present(df, ["batch_size","contract.training.batch_size"]),
        lr = first_present(df, ["lr","contract.training.lr"]),
        weight_decay = first_present(df, ["weight_decay","contract.training.weight_decay"]),
        optimizer = first_present(df, ["optimizer","contract.training.optimizer"]),
        scheduler = first_present(df, ["scheduler","contract.training.scheduler"]),
        dropout = first_present(df, ["dropout","contract.model.dropout"]),
        sample_rate = first_present(df, ["sample_rate","contract.data.sample_rate"]),
        # two-phase / unfreezing policy:
        two_phase_training = first_present(df, ["two_phase_training","contract.training.two_phase_training"]),
        phase1_epochs = first_present(df, ["phase1_epochs","contract.training.phase1_epochs"]),
        head_lr = first_present(df, ["head_lr","contract.training.head_lr"]),
        backbone_lr = first_present(df, ["backbone_lr","contract.training.backbone_lr"]),
        # class imbalance / extras:
        use_focal_loss = first_present(df, ["use_focal_loss","contract.training.use_focal_loss"]),
        use_class_weights = first_present(df, ["use_class_weights","contract.training.use_class_weights"]),
        focal_gamma = first_present(df, ["focal_gamma","contract.training.focal_gamma"]),
        data_ORP = first_present(df, ["data_ORP","contract.data.data_ORP"]),
        head_type = first_present(df, ["head_type","contract.model.head_type"]),
        use_amp = first_present(df, ["use_amp","contract.training.use_amp"]),
        preprocess = first_present(df, ["preprocess","contract.data.preprocess"]),
        # ICL toggles (usually none; included if logged explicitly)
        icl_mode = first_present(df, ["icl_mode"]),
        icl_hidden = first_present(df, ["icl_hidden"]),
        icl_layers = first_present(df, ["icl_layers"]),
    )

    if subj_col is None or scheme_col is None:
        raise ValueError("CSV must contain subject_id and label_scheme/config columns.")

    # normalize
    df = df.rename(columns={subj_col:"subject_id", scheme_col:"label_scheme"})
    if state_col: df = df[df[state_col]=="finished"].copy()

    df["task_config"] = df["label_scheme"].map({"5c":"5-class","4c-v0":"4-class","4c-v1":"4-class"})

    # paired status by presence
    pres = (df.dropna(subset=["task_config","subject_id"])
              .groupby(["subject_id","task_config"]).size().reset_index(name="n"))
    have_5c = set(pres.loc[pres.task_config=="5-class","subject_id"])
    have_4c = set(pres.loc[pres.task_config=="4-class","subject_id"])
    need_4c = sorted(list(have_5c - have_4c))

    print(f"Paired N right now: {len(have_5c & have_4c)}")
    print(f"Subjects missing 4-class: {need_4c}\n")

    if not need_4c:
        print("Nothing to launch.")
        return

    print("=== Run plan for missing 4-class (Light = N1+N2) ===\n")

    for sid in need_4c:
        runs = df[(df.subject_id==sid) & (df.task_config=="5-class")]
        if runs.empty:
            print(f"- {sid}: no 5-class runs found; skipping.\n")
            continue

        # gather values to mirror (mode across that subject's 5c rows)
        vals = {k: (most_common(runs[v]) if v else None) for k,v in cols.items()}
        seeds = []
        if cols["seed"]:
            # support 'seeds' comma-list or scalar 'seed'
            try:
                if runs[cols["seed"]].astype(str).str.contains(",").any():
                    all_seeds = []
                    for s in runs[cols["seed"]].dropna().astype(str):
                        all_seeds += [int(x) for x in s.split(",") if x.strip()!=""]
                    seeds = sorted(set(all_seeds))
                else:
                    seeds = uniq_sorted(runs[cols["seed"]], cast=int)
            except Exception:
                seeds = uniq_sorted(runs[cols["seed"]])

        if args.topk_seeds: seeds = seeds[:args.topk_seeds]

        # pretty summary
        print(f"Subject {sid} (mirror from 5c):")
        print(f"  n_train_subjects : {vals['n_train_subjects']}")
        if vals["train_subjects_yaml"]: print(f"  train_subjects_yaml: {vals['train_subjects_yaml']}")
        if vals["datasets"]:            print(f"  datasets         : {vals['datasets']}")
        if seeds:                       print(f"  seeds            : {','.join(map(str,seeds))}")
        for k in ["epochs","batch_size","lr","weight_decay","optimizer","scheduler",
                  "two_phase_training","phase1_epochs","head_lr","backbone_lr",
                  "use_focal_loss","use_class_weights","focal_gamma","data_ORP",
                  "head_type","use_amp","preprocess","dropout","sample_rate",
                  "icl_mode","icl_hidden","icl_layers"]:
            if vals[k] is not None: print(f"  {k:17}: {vals[k]}")
        print("")

        # ---- compose minimal command line ----
        cmd = [
            f"python3 {args.script}",
            f"--downstream_dataset {args.downstream_dataset}",
            f"--datasets_dir {args.datasets_dir}",
            f"--use_pretrained_weights True",
            f"--model_dir {args.model_dir}",
            f"--label_scheme {args.label_4c}",
            f"--num_of_classes 4",
            f"--label_mapping_version v1",
            f"--test_subject {sid}",
            "--tune",
            f"--run_name granularity_pairing_4c_{sid}"
        ]
        # always include T* (fairness), datasets/train pool, seeds if known
        add_flag(cmd, "n_train_subjects", vals["n_train_subjects"])
        add_flag(cmd, "train_subjects_yaml", vals["train_subjects_yaml"])
        add_flag(cmd, "datasets", vals["datasets"])
        if seeds: cmd.append(f"--seeds {','.join(map(str,seeds))}")

        # include knobs only if they differ from script defaults
        for k in ["epochs","batch_size","lr","weight_decay","optimizer","scheduler",
                  "dropout","sample_rate","phase1_epochs","head_lr","backbone_lr",
                  "focal_gamma","data_ORP","head_type","icl_mode","icl_hidden","icl_layers"]:
            v = vals[k]
            if v is None: continue
            d = TRAIN_DEFAULTS.get(k, object())
            try:
                # numeric compare tolerant to string storage
                if isinstance(d,(int,float)) and isinstance(v,str):
                    v = float(v) if "." in v else int(v)
            except Exception:
                pass
            add_flag(cmd, k, v, default_sentinel=d)

        # booleans
        for k in ["two_phase_training","use_focal_loss","use_class_weights","use_amp","preprocess"]:
            v = vals[k]
            if v is None: continue
            # values may be 'True'/'False' strings
            if isinstance(v,str): v = v.lower() in ["1","true","t","yes","y"]
            d = TRAIN_DEFAULTS.get(k, False)
            if bool(v) != bool(d):
                add_flag(cmd, k, True, boolean=True)

        print("CMD:")
        print("  " + " \\\n  ".join(cmd) + "\n")

if __name__ == "__main__":
    main()
