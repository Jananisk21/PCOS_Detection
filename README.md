# PCOS Detection from Ultrasound Sonography Images

Working implementation of the pipeline proposed in the base paper:

> *A Review on Machine Learning, Deep Learning, and Ensemble Transfer
> Learning Approaches with Hyperparameter Optimization for Polycystic
> Ovary Syndrome (PCOS) Detection from Ultrasound Sonography Images*
> (Nalawade & Gupta)

The review's central proposal is the **EffiDenseGenOp** framework:
Fuzzy-Inference-System image enhancement → an **EfficientNetB7 +
DenseNet201 ensemble** → **Genetic-Algorithm** hyperparameter tuning →
Grad-CAM explainability. This project implements every stage of that
pipeline end-to-end.

## Pipeline

```
Ultrasound image
   │
   ▼
1. Fuzzy Inference System (src/fuzzy_enhancement.py)
   Gaussian/Triangular membership functions estimate local noise level,
   then blend a denoised and a sharpened version of the image per-pixel.
   │
   ▼
2. Preprocessing (src/preprocessing.py)
   Resize, RGB conversion, augmentation.
   │
   ▼
3. Ensemble model (src/models.py)
   EfficientNetB7 + DenseNet201 (ImageNet-pretrained, frozen) → feature
   concatenation → tunable dense classifier head → sigmoid PCOS/notPCOS.
   │
   ▼
4. Genetic Algorithm search (src/genetic_optimizer.py)
   Searches dense-layer width, dropout rate, learning rate, and optimizer
   choice to maximize validation accuracy/AUC.
   │
   ▼
5. Evaluation (evaluate.py)
   Accuracy, AUC-ROC, confusion matrix, classification report.
   │
   ▼
6. Explainability (src/gradcam.py) + Demo UI (app.py)
   Grad-CAM heatmaps highlighting follicle regions; Streamlit app for
   interactive demonstration.
```

## Project layout

```
pcos_detection/
├── src/
│   ├── fuzzy_enhancement.py   # Fuzzy Inference System (noise-aware enhancement)
│   ├── preprocessing.py       # resize/normalize/augment
│   ├── dataset.py             # dataset loading + train/val/test split
│   ├── models.py              # EfficientNetB7 + DenseNet201 ensemble
│   ├── genetic_optimizer.py   # from-scratch Genetic Algorithm
│   └── gradcam.py             # Grad-CAM explainability
├── generate_synthetic_dataset.py  # demo dataset generator (see note below)
├── train.py                   # full training pipeline (GA search + final fit)
├── evaluate.py                # test-set evaluation + plots
├── predict.py                 # single-image CLI inference + Grad-CAM
├── app.py                     # Streamlit demo interface
├── requirements.txt
└── data/raw/{PCOS,notPCOS}/   # put your dataset images here
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get a dataset

Use any ovarian-ultrasound PCOS dataset (e.g. the public Kaggle "PCOS
Detection using Ultrasound Images" dataset referenced by the paper).
Arrange it as:

```
data/raw/
    PCOS/       *.png / *.jpg
    notPCOS/    *.png / *.jpg
```

**No internet access in this sandbox** meant the real Kaggle dataset
could not be downloaded here. A synthetic stand-in dataset generator is
included purely so the full pipeline can be demonstrated end-to-end:

```bash
python generate_synthetic_dataset.py --n_per_class 120
```

This draws speckle-noised "ovary" images: `notPCOS` images have a few
large sparse follicles; `PCOS` images have many small follicles arranged
in a peripheral ring (mimicking the real "string of pearls" sign).
**Replace this with the real dataset for meaningful results** — just
drop real images into the same two folders, no code changes needed.

## 2. Train

```bash
python train.py --data_dir data/raw --img_size 96 \
    --ga_population 6 --ga_generations 3 --ga_epochs 2 --final_epochs 8
```

This runs the GA search (training several quick candidate models),
picks the best hyperparameters, retrains fully, and saves:
- `saved_models/pcos_effidensegenop_model.keras`
- `saved_models/ga_best_hyperparameters.json`
- `saved_models/test_split.npz` (held-out test set for evaluation)

Increase `--img_size` (224 is standard for these backbones), `--ga_population`,
`--ga_generations`, and `--final_epochs` for a real run on a GPU machine.

**On pretrained weights:** EfficientNetB7/DenseNet201 ImageNet weights
are downloaded automatically by Keras on first use. This sandbox's
network policy blocks that download, so here the backbones silently
fall back to random initialization (a warning is printed) — the code
and pipeline are fully correct, but accuracy numbers produced *inside
this sandbox* reflect that limitation, not the model design. Running
`train.py` on a normal machine will fetch the real ImageNet weights
automatically.

## 3. Evaluate

```bash
python evaluate.py
```

Prints/saves accuracy, AUC-ROC, confusion matrix, and ROC curve to `outputs/`.

## 4. Predict on a single image (with Grad-CAM)

```bash
python predict.py --image path/to/scan.png
```

## 5. Interactive demo

```bash
streamlit run app.py
```

Upload an ultrasound image and see: original → fuzzy-enhanced version →
Grad-CAM overlay → predicted class and confidence.

## 6. Ablation study & optimizer comparison (for a paper / thesis)

`run_ablation_study.py` is the intended source of a paper's core results
table and convergence figure. It isolates the contribution of each
pipeline component and compares three hyperparameter-search algorithms
— the base paper's Genetic Algorithm plus Particle Swarm Optimization
and a Grey Wolf Optimizer (both explicitly named as future work in the
base paper's Section VI), all searching the *same* encoded
hyperparameter space (`src/hyperparam_space.py`) for a fair comparison.

```bash
python run_ablation_study.py --data_dir data/raw --img_size 128 \
    --seeds 42 43 44 --search_population 8 --search_iterations 4
```

Runs, across every seed: `single_efficientnet`, `single_densenet`,
`ensemble_no_fuzzy`, `ensemble_default` (no search), `ensemble_ga`,
`ensemble_pso`, `ensemble_gwo`. Outputs to `outputs/ablation/`:

- `ablation_results.csv` — every individual run's accuracy/AUC
- `ablation_summary.md` — a mean ± std markdown table, ready to paste into a paper
- `optimizer_convergence.png` — GA vs. PSO vs. GWO best-fitness-per-iteration

Use more seeds (5+) and full `--search_population`/`--search_iterations`
for publication-quality statistics; the defaults above are for a quick
CPU sanity check.

`train.py` also accepts `--optimizer {ga,pso,gwo}` if you just want to
train a single final model with a specific search algorithm rather than
running the full ablation.

## Notes / scope

- This is a research/demo implementation for coursework and
  experimentation, **not a medical diagnostic tool**.
- Kept deliberately close to what the paper describes — no extra
  unrelated features were added (no additional datasets, tasks, or
  model types beyond the ensemble + GA + fuzzy-enhancement + Grad-CAM
  pipeline the review centers on).
- Backbones are frozen (standard transfer-learning practice for a
  ~200–300 image dataset); optionally unfreeze the top blocks of each
  backbone for fine-tuning once you have a larger dataset and a GPU.
