"""
End-to-end training script for PCOS detection, implementing the full
pipeline proposed in the base paper:

    raw ultrasound images
        -> Fuzzy Inference System noise-aware enhancement
        -> resize / normalize
        -> EfficientNetB7 + DenseNet201 frozen feature extraction (cached
           once, since the backbones never change during training)
        -> Genetic-Algorithm search over classifier-head hyperparameters
           (dense width, dropout, learning rate, optimizer)
        -> final training of the best configuration on the full train set
        -> evaluation on a held-out test set (see evaluate.py)

Usage:
    python train.py --data_dir data/raw --img_size 128 \
        --ga_population 8 --ga_generations 4 --ga_epochs 15 --final_epochs 30

Notes on scale: EfficientNetB7 + DenseNet201 are large backbones -- pass
--backbone_variant lite (EfficientNetB0 + DenseNet121, same pipeline) for
a fast CPU smoke test; use the default --backbone_variant full for
research-grade results, ideally on a GPU with --img_size 224.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from src.dataset import load_dataset, train_val_test_split
from src.optimizers import run_search
from src.models import build_classifier_head, build_feature_extractor, combine_extractor_and_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--use_fuzzy_enhancement", action="store_true", default=True)
    parser.add_argument("--ga_population", type=int, default=8,
                         help="population/swarm/pack size for the search algorithm")
    parser.add_argument("--ga_generations", type=int, default=4,
                         help="number of iterations for the search algorithm")
    parser.add_argument("--optimizer", choices=["ga", "pso", "gwo"], default="ga",
                         help="hyperparameter search algorithm: Genetic Algorithm (base paper), "
                              "Particle Swarm Optimization, or Grey Wolf Optimizer")
    parser.add_argument("--ga_epochs", type=int, default=15,
                         help="quick epochs used per candidate during GA search "
                              "(cheap: the head is a small dense net over cached features)")
    parser.add_argument("--final_epochs", type=int, default=30)
    parser.add_argument("--try_pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", dest="try_pretrained", action="store_false")
    parser.add_argument("--backbone_variant", choices=["full", "lite"], default="full",
                         help="'full' = EfficientNetB7+DenseNet201 (paper-accurate, use for "
                              "real runs/GPU). 'lite' = EfficientNetB0+DenseNet121 (same "
                              "pipeline, fast CPU smoke-test).")
    parser.add_argument("--output_dir", default="saved_models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1/5 — Loading dataset and applying Fuzzy-Inference-System enhancement")
    print("=" * 70)
    X, y = load_dataset(args.data_dir, img_size=args.img_size,
                         use_fuzzy_enhancement=args.use_fuzzy_enhancement, seed=args.seed)
    print(f"Loaded {len(X)} images -> {np.bincount(y)} (notPCOS, PCOS)")

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, seed=args.seed)
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    input_shape = (args.img_size, args.img_size, 3)

    print("\n" + "=" * 70)
    print(f"STEP 2/5 — Extracting frozen EfficientNet+DenseNet ({args.backbone_variant}) "
          f"features (computed once, then cached and reused)")
    print("=" * 70)
    feature_extractor, _, _ = build_feature_extractor(
        input_shape=input_shape, backbone_variant=args.backbone_variant,
        try_pretrained=args.try_pretrained,
    )
    feature_dim = feature_extractor.output_shape[-1]
    print(f"Fused feature dimension: {feature_dim}")

    F_train = feature_extractor.predict(X_train, batch_size=16, verbose=1)
    F_val = feature_extractor.predict(X_val, batch_size=16, verbose=1)
    F_test = feature_extractor.predict(X_test, batch_size=16, verbose=1)

    print("\n" + "=" * 70)
    print(f"STEP 3/5 — {args.optimizer.upper()} search over classifier hyperparameters")
    print("=" * 70)

    def fitness_fn(individual):
        tf.keras.backend.clear_session()
        head = build_classifier_head(
            feature_dim=feature_dim,
            dense_units=individual["dense_units"],
            dropout_rate=individual["dropout_rate"],
            learning_rate=individual["learning_rate"],
            optimizer_name=individual["optimizer_name"],
        )
        history = head.fit(
            F_train, y_train,
            validation_data=(F_val, y_val),
            epochs=args.ga_epochs,
            batch_size=16,
            verbose=0,
        )
        val_acc = history.history["val_accuracy"][-1]
        val_auc = history.history["val_auc"][-1]
        return 0.6 * val_acc + 0.4 * val_auc

    best_individual, best_fitness, search_history = run_search(
        args.optimizer,
        fitness_fn,
        population_size=args.ga_population,
        generations=args.ga_generations,
        seed=args.seed,
    )
    print(f"\nBest hyperparameters found by {args.optimizer.upper()}: {best_individual}")
    print(f"Best fitness (0.6*val_acc + 0.4*val_auc): {best_fitness:.4f}")

    with open(os.path.join(args.output_dir, "search_best_hyperparameters.json"), "w") as f:
        json.dump({"optimizer": args.optimizer, "best_individual": best_individual,
                   "best_fitness": best_fitness, "history": search_history,
                   "backbone_variant": args.backbone_variant,
                   "img_size": args.img_size}, f, indent=2)

    print("\n" + "=" * 70)
    print("STEP 4/5 — Final training with GA-optimized hyperparameters")
    print("=" * 70)
    tf.keras.backend.clear_session()
    feature_extractor, _, _ = build_feature_extractor(
        input_shape=input_shape, backbone_variant=args.backbone_variant,
        try_pretrained=args.try_pretrained,
    )
    final_head = build_classifier_head(
        feature_dim=feature_dim,
        dense_units=best_individual["dense_units"],
        dropout_rate=best_individual["dropout_rate"],
        learning_rate=best_individual["learning_rate"],
        optimizer_name=best_individual["optimizer_name"],
    )
    final_head.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                          patience=6, restore_best_weights=True),
    ]
    final_head.fit(
        F_train, y_train,
        validation_data=(F_val, y_val),
        epochs=args.final_epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n" + "=" * 70)
    print("STEP 5/5 — Assembling end-to-end model (raw image in -> probability out) and saving")
    print("=" * 70)
    full_model = combine_extractor_and_head(feature_extractor, final_head)

    model_path = os.path.join(args.output_dir, "pcos_effidensegenop_model.keras")
    full_model.save(model_path)
    print(f"Saved final end-to-end model to {model_path}")

    np.savez(os.path.join(args.output_dir, "test_split.npz"), X_test=X_test, y_test=y_test)
    print(f"Saved held-out test split to {args.output_dir}/test_split.npz")

    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump({"img_size": args.img_size, "backbone_variant": args.backbone_variant}, f, indent=2)

    print("\nDone. Run `python evaluate.py` to score the held-out test set.")


if __name__ == "__main__":
    main()
