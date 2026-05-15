# Malaria cell classifier, in the browser

A small convolutional neural network that looks at a stained blood-cell photo
and predicts infected or uninfected, with an uncertainty interval. The model
runs entirely in the visitor's browser (ONNX Runtime Web). No image is uploaded
anywhere.

Live demo: `site/index.html` (static, no build step).

## The problem

Malaria diagnosis from thin blood smears means a microscopist scanning cells
one by one. A classifier that flags likely-infected cells can speed up
screening. This project takes two pre-trained CNNs from an old class project,
evaluates them honestly on public data, keeps the better one, and ships it as
a web demo with uncertainty attached to every prediction.

## The data

The NIH malaria cell image set (Rajaraman et al., 2018): 27,558 segmented
thin-blood-smear cell photos, half parasitized, half uninfected, collected by
the U.S. National Library of Medicine with Mahidol-Oxford and Chittagong
Medical College Hospital. I used the official NLM mirror.

File names carry only a cell ID. There are no parasite life-cycle stage
labels, so I could not do error analysis by stage. I did a morphology-based
error analysis instead and say so where it matters.

## Methods

I evaluated both shipped models (`malaria_cnn.h5`, `my_model.h5`) on the full
public set, since the original repo records no train/test split.

- **Preprocessing check.** The repo's inference script feeds raw 0-255 BGR
  pixels. I tried four variants (BGR/RGB x raw/[0,1]) on 2,000 images.
  Raw pixels saturate `my_model` (it outputs the same answer for everything).
  RGB scaled to [0,1] gives 94.9%. The shipped weights were clearly trained on
  scaled RGB, not on what the repo's script feeds them.
- **Winner by measured accuracy.** `my_model` (50x50 input) vs `malaria_cnn`
  (32x32 input). No contest; see results.
- **Bootstrap 95% CIs** (2,000 resamples) for accuracy, sensitivity,
  specificity, precision, F1, AUC, and every confusion matrix count. Point
  estimates without intervals hide the sampling noise.
- **Test-time augmentation for uncertainty.** Each cell is shown to the model
  in 8 geometric views (4 rotations, with and without horizontal flip). I
  report the mean P(infected), its SD, and a 95% interval. Wide interval
  means the model is unsure about that cell.
- **Calibration check.** Expected calibration error over 15 bins plus a
  reliability diagram, because a stated 0.9 should mean right 90% of the time.
- **Error analysis.** Compared misclassified cells to correctly classified
  ones on brightness, contrast, saturation, and redness, and plotted the
  worst false positives and false negatives.

I converted the Keras weights to ONNX with a small hand-written converter
(`h5_to_onnx.py`) and verified it against an independent NumPy forward pass
of each network (max difference 0.0 for `my_model`). The analysis itself runs
on the ONNX files, so the README numbers come from the exact artifact shipped
in `site/`.

## Results (n = 27,558)

`my_model` wins clearly.

| metric | my_model | 95% CI | malaria_cnn | 95% CI |
|---|---|---|---|---|
| accuracy | 0.932 | 0.929-0.935 | 0.593 | 0.588-0.599 |
| sensitivity | 0.929 | 0.925-0.933 | 0.992 | 0.990-0.993 |
| specificity | 0.935 | 0.930-0.939 | 0.195 | 0.189-0.202 |
| AUC | 0.958 | 0.955-0.960 | 0.780 | 0.775-0.785 |
| Brier | 0.071 | - | 0.342 | - |

Confusion matrix, `my_model`: 12,877 TN, 902 FP, 975 FN, 12,804 TP.

93.2% lands right next to the published 92.7% for a small custom CNN on this
set (Rajaraman et al.), which is what an honest result here looks like.

Calibration: expected calibration error 0.103. The model is under-confident
in the middle range (cells scored 0.4-0.7 are infected more often than
claimed). That is the safer direction to be wrong in for screening.

TTA uncertainty (2,000 cells): mean SD across the 8 views is 0.055, mean 95%
interval width 0.177. Missed cells have slightly wider intervals (0.201) than
correct ones (0.175), so the interval carries a weak signal about which
predictions to distrust.

Errors: 1,877 misses. They run slightly darker (brightness 0.446 vs 0.470)
and more saturated (0.200 vs 0.173) than correct calls. See
`figures/errors_false_negatives.png` and `figures/errors_false_positives.png`.

Caveat: the original training split is unknown, so the models may have seen
some of these images in training. Read the numbers as an upper bound on
held-out performance, not a generalization guarantee. This is a research
demo, not a medical device.

## How the web demo works

`site/` is plain HTML + JS, no build step.

1. `my_model.onnx` (2.4 MB) loads from a relative path with ONNX Runtime Web
   from the jsDelivr CDN.
2. Your image is resized to 50x50 in a canvas, converted to RGB float32 in
   [0, 1], and fed to the model in 8 rotated/flipped views.
3. The page shows Infected/Uninfected, the mean P(infected), and the 95%
   interval, plus a per-view table. The 4 sample cells in `site/samples/`
   have precomputed results in `results.json` (true labels included); the
   page recomputes them live when you click.

## Reproduce

```
python -m venv .venv && source .venv/bin/activate
pip install onnx onnxruntime h5py numpy pillow scikit-learn matplotlib
python h5_to_onnx.py            # .h5 -> models/*.onnx
python analyze.py --quick       # preprocessing sanity check
python analyze.py --full        # full eval, figures, metrics.json
```

## Files

- `analyze.py`, `analysis_lib.py`: evaluation code
- `analysis.ipynb`: executed notebook version of the analysis
- `h5_to_onnx.py`: Keras weights to ONNX converter
- `figures/`: confusion matrices, ROC, calibration, TTA spread, error montages
- `metrics.json`, `error_report.json`: all computed numbers
- `REPORT.md`: method justifications and full results
- `site/`: the browser demo

## Sources

- Rajaraman, S. et al. (2018). Pre-trained convolutional neural networks as
  feature extractors toward improved malaria parasite detection in thin blood
  smear images. PeerJ 6:e4568. (Dataset and the 92.7% / 95.9% benchmarks.)
- Ayhan, M. S. and Berens, P. (2018). Test-time data augmentation for
  estimation of heteroscedastic aleatoric uncertainty in deep neural networks.
  MIDL 2018. (TTA uncertainty.)
- Guo, C. et al. (2017). On calibration of modern neural networks. ICML 2017.
  (Expected calibration error.)
- Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
  Chapman & Hall. (Bootstrap confidence intervals.)
