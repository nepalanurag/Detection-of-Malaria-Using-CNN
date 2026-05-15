# Malaria cell classification: evaluation report

## What I did and why

I took the two CNNs from the original project (`malaria_cnn.h5`, `my_model.h5`) and
evaluated them properly on the public NIH malaria cell image set instead of trusting
the training-time numbers. The original repo records no train/test split, so I report
everything on the full public set and say so plainly where it matters.

### Data

The NIH malaria cell image dataset: 27,558 segmented thin-blood-smear cell PNGs,
13,779 parasitized and 13,779 uninfected, collected by the U.S. National Library of
Medicine with Mahidol-Oxford Tropical Medicine Research Unit and Chittagong Medical
College Hospital, and released with Rajaraman et al. (2018). I downloaded it from the
official NLM mirror (data.lhncbc.nlm.nih.gov). File names carry only a cell ID, no
parasite life-cycle stage, so a stage-by-stage error analysis is not possible with
this data. I say this outright rather than guessing stages from morphology.

Rajaraman et al. reported 92.7% accuracy with a small custom CNN trained from scratch
and 95.9% (sensitivity 94.7%) with a ResNet50 used as a feature extractor. Later work
on the same set reaches 96-99% with transfer learning. Any honest small-CNN result on
this data should land roughly in the low-to-mid 90s; far above that would smell of
train/test leakage, far below would mean something is broken.

### Method choices

- **Evaluate both models, keep the better.** The two nets differ in input size
  (32x32 vs 50x50), depth, and activation style. Picking by measured accuracy on the
  same data is the only defensible selection rule. I chose accuracy because the
  classes are balanced, so it is not misleading here.
- **Preprocessing recovery.** The repo's own inference script feeds raw 0-255 pixels
  in BGR channel order (OpenCV default). I checked four preprocessing variants on a
  2,000-image subset. Raw pixels saturate `my_model` (it outputs [1, 0] for every
  image, ~50% accuracy); scaling to [0, 1] restores it to 94.9% with RGB order
  slightly ahead of BGR. So the shipped weights do not match the repo's inference
  script; I use RGB / 255, the variant the weights were evidently trained with.
  `malaria_cnn` stays near chance under all four variants (best 58.6%), so no
  preprocessing rescues it. The subset check is in the notebook so the choice is
  auditable.
- **Bootstrap percentile intervals** (Efron and Tibshirani, 1993) with 2,000
  resamples for accuracy, sensitivity, specificity, F1, AUC, and each confusion
  matrix count. Metrics like these have no trustworthy closed-form standard error,
  and the percentile bootstrap makes no distributional assumptions. I report
  intervals, not just point estimates, because a bare 96% means little without its
  sampling noise.
- **Test-time augmentation for uncertainty** (Ayhan and Berens, 2018): each image is
  shown to the model in 8 geometric views (4 rotations, with and without horizontal
  flip). I report the mean predicted probability, its SD, and a 95% interval
  (mean +/- 1.96 SD). This estimates input-dependent (aleatoric) uncertainty: cells
  the model finds ambiguous get wide intervals. I use deterministic geometric views
  rather than random color jitter so the demo is reproducible in the browser.
- **Calibration check** with expected calibration error over 15 equal-width bins
  (Guo et al., 2017), plus a reliability diagram. A screening model whose 0.9 does
  not mean "right 90% of the time" is dangerous; ECE measures exactly that gap.
- **Error analysis by morphology, not by stage.** With no stage labels available, I
  compared misclassified cells against correctly classified ones on simple,
  interpretable image properties (brightness, contrast, saturation, redness) and
  plotted the worst false positives and false negatives. The goal is to find
  failure patterns a microscopist would recognize, not to over-claim.

### Honest caveats

- The original training split is unknown, so these models may have seen some of
  these images during training. Treat the headline numbers as an upper bound on
  true held-out performance, not as a generalization guarantee.
- This is a research demo, not a medical device. It classifies single cropped
  cells, which is only one step of real malaria diagnosis.

## Results (n = 27,558)

`my_model` (50x50 input, 3 conv layers + dense 512, RGB / 255) is the clear
winner. `malaria_cnn` (32x32 input, sigmoid activations) is barely above chance
and its 0.5 threshold is badly placed (sensitivity 99%, specificity 20%).

| metric | my_model | 95% CI | malaria_cnn | 95% CI |
|---|---|---|---|---|
| accuracy | 0.932 | 0.929-0.935 | 0.593 | 0.588-0.599 |
| sensitivity | 0.929 | 0.925-0.933 | 0.992 | 0.990-0.993 |
| specificity | 0.935 | 0.930-0.939 | 0.195 | 0.189-0.202 |
| precision | 0.934 | 0.930-0.938 | 0.552 | 0.546-0.558 |
| F1 | 0.932 | 0.929-0.935 | 0.709 | 0.704-0.714 |
| AUC | 0.958 | 0.955-0.960 | 0.780 | 0.775-0.785 |
| Brier score | 0.071 | - | 0.342 | - |

Confusion matrix for `my_model` (bootstrap 95% CIs in brackets):
TN 12,877 [12,725-13,032], FP 902 [843-961], FN 975 [917-1,036],
TP 12,804 [12,644-12,963].

The 93.2% sits right next to Rajaraman et al.'s 92.7% for a small custom CNN,
which is what a result on this data should look like. The intervals are tight
because n is large; with 27,558 images even the third decimal is pinned down.

**Calibration.** Expected calibration error is 0.103 over 15 bins. The
reliability diagram shows the model is under-confident in the middle: cells it
scores 0.4-0.7 are infected more often than it claims. A screening tool should
not overstate its certainty, but here the bias runs the other way, which is the
safer direction to be wrong in.

**TTA uncertainty** (2,000 images, 8 views each). Mean SD of P(infected) across
views is 0.055; mean 95% interval width is 0.177. Correctly classified cells
average 0.175 wide, misclassified ones 0.201: the interval carries a weak
signal about which predictions to distrust, but the SD alone does not separate
them (0.055 vs 0.053). Uncertainty helps, it does not solve error detection.

**Error analysis.** 1,877 misses out of 27,558. Misclassified cells are a touch
darker (mean brightness 0.446 vs 0.470) and more saturated (0.200 vs 0.173)
than correctly classified ones; contrast and redness are the same. The montage
figures show the worst false negatives and false positives. No life-cycle stage
labels exist in this data, so this morphology comparison is as far as honest
error analysis can go.

## References

- Rajaraman, S., Antani, S. K., Poostchi, M., Silamut, K., Hossain, M. A.,
  Maude, R. J., Jaeger, S., and Thoma, G. R. (2018). Pre-trained convolutional
  neural networks as feature extractors toward improved malaria parasite detection
  in thin blood smear images. PeerJ 6:e4568.
- Ayhan, M. S. and Berens, P. (2018). Test-time data augmentation for estimation
  of heteroscedastic aleatoric uncertainty in deep neural networks. MIDL 2018.
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of
  modern neural networks. ICML 2017.
- Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap.
  Chapman & Hall.
