"""
Ablation study and optimizer-comparison experiment runner.

This is the intended source of the paper's core results table and
convergence figure. It runs a fixed set of configurations, each across
multiple random seeds, and reports mean +/- std accuracy / AUC-ROC on a
held-out test set -- rather than a single run's numbers.

Configurations run (see CONFIGS below):
    1. single_efficientnet          - EfficientNet backbone alone, default head hyperparameters
    2. single_densenet              - DenseNet backbone alone, default head hyperparameters
    3. ensemble_no_fuzzy            - both backbones, ensemble, but WITHOUT fuzzy enhancement
    4. ensemble_default             - both backbones, ensemble, WITH fuzzy enhancement, default hyperparameters (no search)
    5. ensemble_ga                  - full pipeline, classifier head tuned with a Genetic Algorithm
    6. ensemble_pso                 - full pipeline, classifier head tuned with Particle Swarm Optimization
    7. ensemble_gwo                 - full pipeline, classifier head tuned with a Grey Wolf Optimizer

This isolates: (a) the contribution of ensembling vs. a single backbone,
(b) the contribution of the fuzzy-enhancement preprocessing stage, and
(c) which of the three optimizers finds the best classifier-head
hyperparameters -- directly answering the base paper's own suggestion to
explore optimizers beyond the Genetic Algorithm.

Usage:
    python run_ablation_study.py --data_dir data/raw --img_size 128 \
        --seeds 42 43 44 --search_population 8 --search_iterations 4

Outputs (under --output_dir, default "outputs/ablation"):
    - ablation_results.csv       raw per-run results
    - ablation_summary.md        mean +/- std table, ready to paste into the paper
    - optimizer_convergence.png  GA vs PSO vs GWO best-fitness-per-iteration
"""

import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score

from src.dataset import load_dataset, train_val_test_split
from src.models import build_classifier_head, build_feature_extractor, combine_extractor_and_head
from src.optimizers import run_search

DEFAULT_HYPERPARAMS = {
    "dense_units": 256, "dropout_rate": 0.4,
    "learning_rate": 1e-4, "optimizer_name": "adam",
}


def get_features(input_shape, backbone_variant, try_pretrained, X_train, X_val, X_test):
    tf.keras.backend.clear_session()
    extractor, _, _ = build_feature_extractor(
        input_shape=input_shape, backbone_variant=backbone_variant, try_pretrained=try_pretrained)
    F_train = extractor.predict(X_train, batch_size=16, verbose=0)
    F_val = extractor.predict(X_val, batch_size=16, verbose=0)
    F_test = extractor.predict(X_test, batch_size=16, verbose=0)
    return extractor, F_train, F_val, F_test


def train_and_score(feature_dim, hyperparams, F_train, y_train, F_val, y_val,
                     F_test, y_test, epochs):
    tf.keras.backend.clear_session()
    head = build_classifier_head(feature_dim=feature_dim, **hyperparams)
    head.fit(F_train, y_train, validation_data=(F_val, y_val),
              epochs=epochs, batch_size=16, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_auc", mode="max", patience=6, restore_best_weights=True)])
    y_prob = head.predict(F_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present in a tiny demo test split
    return acc, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--backbone_variant", choices=["full", "lite"], default="full")
    parser.add_argument("--try_pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="try_pretrained", action="store_false")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--search_population", type=int, default=6)
    parser.add_argument("--search_iterations", type=int, default=4)
    parser.add_argument("--search_epochs", type=int, default=10,
                         help="epochs per candidate during hyperparameter search")
    parser.add_argument("--final_epochs", type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/ablation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_shape = (args.img_size, args.img_size, 3)
    results = []
    convergence_curves = {}  # optimizer name -> list of histories (one per seed)

    for seed in args.seeds:
        print("\n" + "#" * 70)
        print(f"# SEED {seed}")
        print("#" * 70)
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # --- Load data twice: with and without fuzzy enhancement ---
        X_fuzzy, y = load_dataset(args.data_dir, img_size=args.img_size,
                                   use_fuzzy_enhancement=True, seed=seed)
        X_plain, _ = load_dataset(args.data_dir, img_size=args.img_size,
                                   use_fuzzy_enhancement=False, seed=seed)

        Xf_train, y_train, Xf_val, y_val, Xf_test, y_test = train_val_test_split(X_fuzzy, y, seed=seed)
        Xp_train, _, Xp_val, _, Xp_test, _ = train_val_test_split(X_plain, y, seed=seed)

        # --- Cache features for every backbone configuration we need ---
        print("Extracting ensemble features (fuzzy-enhanced input)...")
        _, Fens_train, Fens_val, Fens_test = get_features(
            input_shape, args.backbone_variant, args.try_pretrained, Xf_train, Xf_val, Xf_test)

        print("Extracting ensemble features (no fuzzy enhancement)...")
        _, Fnf_train, Fnf_val, Fnf_test = get_features(
            input_shape, args.backbone_variant, args.try_pretrained, Xp_train, Xp_val, Xp_test)

        eff_only = "efficientnetb7" if args.backbone_variant == "full" else "efficientnetb0"
        dense_only = "densenet201" if args.backbone_variant == "full" else "densenet121"

        # Single-backbone feature sets: reuse the ensemble extractor internals by
        # slicing the concatenated feature vector is not exact per-backbone, so
        # we build dedicated single-backbone extractors for a clean ablation.
        from src.models import _build_backbone
        import tensorflow.keras as keras

        def single_backbone_features(name):
            tf.keras.backend.clear_session()
            inp = keras.layers.Input(shape=input_shape)
            backbone = _build_backbone(name, input_shape, args.try_pretrained)
            backbone.trainable = False
            if "efficientnet" in name:
                pre = keras.applications.efficientnet.preprocess_input(inp)
            else:
                pre = keras.applications.densenet.preprocess_input(inp)
            feat = backbone(pre)
            extractor = keras.Model(inp, feat)
            return (extractor.predict(Xf_train, batch_size=16, verbose=0),
                    extractor.predict(Xf_val, batch_size=16, verbose=0),
                    extractor.predict(Xf_test, batch_size=16, verbose=0),
                    extractor.output_shape[-1])

        print("Extracting single-backbone (EfficientNet only) features...")
        Feff_train, Feff_val, Feff_test, eff_dim = single_backbone_features(eff_only)
        print("Extracting single-backbone (DenseNet only) features...")
        Fdn_train, Fdn_val, Fdn_test, dn_dim = single_backbone_features(dense_only)

        ens_dim = Fens_train.shape[-1]

        # --- 1/2: single backbones, default hyperparameters ---
        for cfg_name, (Ftr, Fv, Fte, dim) in [
            ("single_efficientnet", (Feff_train, Feff_val, Feff_test, eff_dim)),
            ("single_densenet", (Fdn_train, Fdn_val, Fdn_test, dn_dim)),
        ]:
            acc, auc = train_and_score(dim, DEFAULT_HYPERPARAMS, Ftr, y_train, Fv, y_val,
                                        Fte, y_test, args.final_epochs)
            results.append({"config": cfg_name, "seed": seed, "accuracy": acc, "auc": auc})
            print(f"[{cfg_name}] seed={seed} acc={acc:.4f} auc={auc:.4f}")

        # --- 3: ensemble without fuzzy enhancement ---
        acc, auc = train_and_score(ens_dim, DEFAULT_HYPERPARAMS, Fnf_train, y_train, Fnf_val,
                                    y_val, Fnf_test, y_test, args.final_epochs)
        results.append({"config": "ensemble_no_fuzzy", "seed": seed, "accuracy": acc, "auc": auc})
        print(f"[ensemble_no_fuzzy] seed={seed} acc={acc:.4f} auc={auc:.4f}")

        # --- 4: ensemble with fuzzy enhancement, default hyperparameters ---
        acc, auc = train_and_score(ens_dim, DEFAULT_HYPERPARAMS, Fens_train, y_train, Fens_val,
                                    y_val, Fens_test, y_test, args.final_epochs)
        results.append({"config": "ensemble_default", "seed": seed, "accuracy": acc, "auc": auc})
        print(f"[ensemble_default] seed={seed} acc={acc:.4f} auc={auc:.4f}")

        # --- 5/6/7: ensemble + GA / PSO / GWO tuned hyperparameters ---
        for opt_name in ["ga", "pso", "gwo"]:
            def fitness_fn(individual, _dim=ens_dim, _Ftr=Fens_train, _Fv=Fens_val):
                acc_i, auc_i = train_and_score(_dim, individual, _Ftr, y_train, _Fv, y_val,
                                                _Fv, y_val, args.search_epochs)
                return 0.6 * acc_i + 0.4 * (0 if np.isnan(auc_i) else auc_i)

            history = []

            def log_cb(it, best_ind, best_fit, _history=history):
                _history.append((it, best_fit))

            best_individual, _, _ = run_search(
                opt_name, fitness_fn, population_size=args.search_population,
                generations=args.search_iterations, seed=seed, verbose=True,
                log_callback=log_cb,
            )
            convergence_curves.setdefault(opt_name, []).append(history)

            acc, auc = train_and_score(ens_dim, best_individual, Fens_train, y_train, Fens_val,
                                        y_val, Fens_test, y_test, args.final_epochs)
            cfg_name = f"ensemble_{opt_name}"
            results.append({"config": cfg_name, "seed": seed, "accuracy": acc, "auc": auc,
                             "best_hyperparameters": json.dumps(best_individual)})
            print(f"[{cfg_name}] seed={seed} acc={acc:.4f} auc={auc:.4f} | {best_individual}")

    # --- Save raw results ---
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, "ablation_results.csv"), index=False)

    # --- Summary table (mean +/- std) ---
    summary = df.groupby("config").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        n_runs=("seed", "count"),
    ).reset_index()

    config_order = ["single_efficientnet", "single_densenet", "ensemble_no_fuzzy",
                     "ensemble_default", "ensemble_ga", "ensemble_pso", "ensemble_gwo"]
    summary["config"] = pd.Categorical(summary["config"], categories=config_order, ordered=True)
    summary = summary.sort_values("config")

    with open(os.path.join(args.output_dir, "ablation_summary.md"), "w") as f:
        f.write("# Ablation Study and Optimizer Comparison Results\n\n")
        f.write(f"Seeds: {args.seeds} | Image size: {args.img_size} | "
                f"Backbone variant: {args.backbone_variant}\n\n")
        f.write("| Configuration | Accuracy (mean ± std) | AUC-ROC (mean ± std) | Runs |\n")
        f.write("|---|---|---|---|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['config']} | {row['accuracy_mean']*100:.2f}% ± "
                     f"{row['accuracy_std']*100:.2f}% | {row['auc_mean']*100:.2f}% ± "
                     f"{row['auc_std']*100:.2f}% | {int(row['n_runs'])} |\n")

    print("\n" + "=" * 70)
    print(open(os.path.join(args.output_dir, "ablation_summary.md")).read())
    print("=" * 70)

    # --- Convergence plot: GA vs PSO vs GWO ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"ga": "#1f77b4", "pso": "#ff7f0e", "gwo": "#2ca02c"}
    for opt_name, histories in convergence_curves.items():
        max_len = max(len(h) for h in histories)
        padded = np.full((len(histories), max_len), np.nan)
        for i, h in enumerate(histories):
            vals = [v for _, v in h]
            padded[i, :len(vals)] = vals
        mean_curve = np.nanmean(padded, axis=0)
        ax.plot(range(1, max_len + 1), mean_curve, label=opt_name.upper(),
                 color=colors.get(opt_name), marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (0.6·accuracy + 0.4·AUC)")
    ax.set_title("Optimizer Convergence Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "optimizer_convergence.png"), dpi=150)

    print(f"\nSaved ablation_results.csv, ablation_summary.md, optimizer_convergence.png "
          f"to {args.output_dir}/")


if __name__ == "__main__":
    main()
