"""Driver for the malaria CNN evaluation.

Usage:
  python analyze.py --quick   # 2000-image sanity check + preprocessing selection
  python analyze.py --full    # full evaluation, bootstrap CIs, TTA, calibration,
                              # error analysis, figures, metrics.json

Keeps CPU use low: TF intra/inter-op threads pinned to 2.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
# onnxruntime's SQLite telemetry store hangs on this machine's filesystem;
# disable it (inference results are unaffected).
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

import argparse
import json
import shutil
import numpy as np

import analysis_lib as L

SRC = os.path.expanduser("~/workspace/diagnostic-demos/sources/Detection-of-Malaria-Using-CNN")
DATA = os.path.expanduser("~/workspace/diagnostic-demos/malaria-cnn-web/data/cell_images")

MODELS = {
    "malaria_cnn": {"file": "malaria_cnn.h5", "size": 32},
    "my_model": {"file": "my_model.h5", "size": 50},
}
PREPROCESS_GRID = [("bgr", 1.0), ("rgb", 1.0), ("rgb", 1.0 / 255.0), ("bgr", 1.0 / 255.0)]


def load_models():
    models = {}
    for name, spec in MODELS.items():
        m = L.ORTModel(f"models/{name}.onnx", threads=2)
        print(f"{name}: onnx loaded, input {spec['size']}x{spec['size']}", flush=True)
        models[name] = m
    return models


def quick(models, items):
    """Sanity check on a 2000-image subset; pick preprocessing per model."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(items), size=2000, replace=False)
    sub = [items[i] for i in sorted(idx)]
    choice = {}
    for name, m in models.items():
        size = MODELS[name]["size"]
        print(f"--- {name} preprocessing check ---", flush=True)
        best = None
        for order, scale in PREPROCESS_GRID:
            prob, y, pcol = L.batch_predict(m, sub, size, order, scale)
            acc = float(((prob >= 0.5).astype(int) == y).mean())
            print(f"  order={order} scale={scale:g} pcol={pcol} acc={acc:.4f}", flush=True)
            if best is None or acc > best[0]:
                best = (acc, order, scale, pcol)
        choice[name] = {"order": best[1], "scale": best[2], "pcol": best[3]}
        print(f"  -> chosen: {choice[name]} (acc {best[0]:.4f})", flush=True)
    with open("preprocess_choice.json", "w") as f:
        json.dump(choice, f, indent=2)
    return choice


def full(models, items, choice):
    os.makedirs("figures", exist_ok=True)
    results, cis = {}, {}
    for name, m in models.items():
        size = MODELS[name]["size"]
        c = choice[name]
        ckpt = f"probs_{name}.npz"
        if os.path.exists(ckpt):
            d = np.load(ckpt)
            prob, y, pcol = d["prob"], d["labels"], int(d["pcol"])
            print(f"=== {name}: loaded checkpoint {ckpt} ===", flush=True)
        else:
            print(f"=== evaluating {name} ({c}) ===", flush=True)
            prob, y, pcol = L.batch_predict(m, items, size, c["order"], c["scale"],
                                            verbose=True)
            assert pcol == c["pcol"], "column detection disagreed with quick run"
            np.savez(ckpt, prob=prob, labels=y, pcol=pcol)
            print(f"saved {ckpt}", flush=True)
        met = L.metrics_from(y, prob)
        ci = L.bootstrap_ci(y, prob)
        results[name] = {"metrics": met, "ci": ci, "prob": prob, "y": y,
                         "preprocess": c}
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                          for k, v in met.items()}, indent=2), flush=True)
        L.plot_confusion(met, ci, f"figures/confusion_{name}.png",
                         f"Confusion matrix, {name} (n={met['n']:,})")
    with open("metrics.json", "w") as f:
        json.dump({n: {"metrics": r["metrics"], "ci": r["ci"],
                       "preprocess": r["preprocess"]}
                   for n, r in results.items()}, f, indent=2)
    L.plot_roc({n: (r["y"], r["prob"]) for n, r in results.items()},
               "figures/roc_both.png")
    # winner = higher accuracy
    winner = max(results, key=lambda n: results[n]["metrics"]["accuracy"])
    print("WINNER:", winner, flush=True)
    w = results[winner]
    # calibration
    ece = L.expected_calibration_error(w["prob"], w["y"])
    print(f"ECE({winner}) = {ece:.4f}", flush=True)
    L.plot_calibration(w["y"], w["prob"], "figures/calibration_winner.png",
                       f"Reliability diagram, {winner}", ece)
    # TTA uncertainty
    tta = L.tta_predict(models[winner], items, MODELS[winner]["size"],
                        w["preprocess"]["order"], w["preprocess"]["scale"])
    L.plot_tta(tta, "figures/tta_spread.png")
    tta_sum = {
        "n": len(tta["mean"]),
        "mean_sd_all": float(tta["sd"].mean()),
        "mean_sd_correct": float(tta["sd"][tta["correct"]].mean()),
        "mean_sd_wrong": float(tta["sd"][~tta["correct"]].mean()),
        "mean_width_all": float((tta["hi"] - tta["lo"]).mean()),
        "mean_width_correct": float((tta["hi"] - tta["lo"])[tta["correct"]].mean()),
        "mean_width_wrong": float((tta["hi"] - tta["lo"])[~tta["correct"]].mean()),
    }
    print("TTA:", json.dumps(tta_sum, indent=2), flush=True)
    np.savez("tta_winner.npz", mean=tta["mean"], sd=tta["sd"], lo=tta["lo"],
             hi=tta["hi"], labels=tta["labels"], correct=tta["correct"],
             paths=np.array(tta["paths"]))
    # error analysis: morphology proxies for errors vs correct
    y, prob = w["y"], w["prob"]
    pred = (prob >= 0.5).astype(int)
    err_idx = np.where(pred != y)[0]
    ok_idx = np.where(pred == y)[0]
    rng = np.random.default_rng(1)
    ok_sub = rng.choice(ok_idx, size=min(2000, len(ok_idx)), replace=False)
    stats_err = L.image_stats([items[i][0] for i in err_idx])
    stats_ok = L.image_stats([items[i][0] for i in ok_sub])
    err_report = {}
    for k in stats_err:
        a, b = stats_err[k], stats_ok[k]
        err_report[k] = {
            "errors_mean": float(a.mean()), "correct_mean": float(b.mean()),
            "errors_sd": float(a.std()), "correct_sd": float(b.std()),
        }
    print("ERROR MORPHOLOGY:", json.dumps(err_report, indent=2), flush=True)
    with open("error_report.json", "w") as f:
        json.dump({"summary": err_report, "n_errors": int(len(err_idx)),
                   "winner": winner, "ece": ece, "tta": tta_sum}, f, indent=2)
    # montages: worst false negatives and false positives
    fn = err_idx[(y[err_idx] == 1)]
    fp = err_idx[(y[err_idx] == 0)]
    fn = fn[np.argsort(prob[fn])]          # most confident misses first
    fp = fp[np.argsort(-prob[fp])]
    L.plot_error_montage([items[i][0] for i in fn], prob[fn], y[fn],
                         "figures/errors_false_negatives.png",
                         "False negatives: truly infected, called clean")
    L.plot_error_montage([items[i][0] for i in fp], prob[fp], y[fp],
                         "figures/errors_false_positives.png",
                         "False positives: truly clean, called infected")
    print("done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    items = L.find_images(DATA)
    n1 = sum(l for _, l in items)
    print(f"dataset: {len(items)} images, {n1} parasitized, "
          f"{len(items) - n1} uninfected", flush=True)
    models = load_models()
    if a.quick or not a.full:
        quick(models, items)
    if a.full:
        choice = {k: {"order": v["order"], "scale": v["scale"], "pcol": v["pcol"]}
                  for k, v in json.load(open("preprocess_choice.json")).items()}
        full(models, items, choice)


if __name__ == "__main__":
    main()
