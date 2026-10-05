# Retired artifacts

`pages_Predict.py` (retired 2026-10-05): the legacy Streamlit prediction app.
It loaded `malaria_cnn.h5` — the model the `web/` analysis rejected (accuracy 0.593, near chance) — and fed raw 0-255 BGR images with no `/255` scaling, which the `web/REPORT.md` shows saturates the model. The shipped demo is the ONNX Runtime Web app in `web/` using `my_model` (accuracy 0.932). This file is kept for reference only.
