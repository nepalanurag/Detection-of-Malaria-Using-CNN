"""Convert the repo's Keras .h5 models to ONNX without TensorFlow.

Why hand-rolled: the two nets are tiny sequential CNNs (conv/pool/flatten/dense
with sigmoid, relu, softmax). Every op maps 1:1 onto ONNX, so a direct weight
translation is exact and auditable. Dropout layers are identity at inference time
and are skipped. The result is cross-checked against tf2onnx output once the TF
environment finishes installing (see REPORT.md).

Keras conventions reproduced exactly:
- Conv2D kernel layout (kH, kW, Cin, Cout) -> ONNX (Cout, Cin, kH, kW)
- padding='valid' -> no pads; padding='same' -> asymmetric TF-style pads
  (for even kernels: extra pad goes bottom/right)
- Dense Y = X @ W + b  ==  ONNX Gemm with transB=0
- Graph runs in NCHW internally; a Transpose at the input accepts NHWC like the
  browser will feed it.
"""
import json
import h5py
import numpy as np
import onnx
from onnx import helper, TensorProto

ACT = {"sigmoid": "Sigmoid", "relu": "Relu", "softmax": "Softmax",
       "linear": None, None: None}


def same_pads(h, w, kh, kw, sh, sw):
    """TF 'same' padding -> ONNX pads [top, left, bottom, right]."""
    out_h = int(np.ceil(h / sh))
    out_w = int(np.ceil(w / sw))
    pad_h = max((out_h - 1) * sh + kh - h, 0)
    pad_w = max((out_w - 1) * sw + kw - w, 0)
    return [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2]


def convert(h5_path, onnx_path, input_hw):
    hf = h5py.File(h5_path, "r")
    cfg = json.loads(hf.attrs["model_config"])
    layers = cfg["config"]["layers"]
    wgrp = hf["model_weights"]

    nodes, inits = [], []
    h = w = input_hw

    def init(name, arr):
        t = helper.make_tensor(name, TensorProto.FLOAT, arr.shape,
                               arr.astype(np.float32).ravel())
        inits.append(t)
        return name

    def weights(layer_name):
        g = wgrp[layer_name]
        # Keras nests params one level deeper under a (sometimes renamed) group
        subs = [k for k in g.keys() if isinstance(g[k], h5py.Group)]
        if len(subs) == 1:
            g = g[subs[0]]
        out = {}
        for k in g.keys():
            # keys look like 'kernel:0' / 'bias:0'
            if isinstance(g[k], h5py.Dataset):
                out[k.split(":")[0]] = np.array(g[k])
        return out

    xin = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                        ["batch", input_hw, input_hw, 3])
    # NHWC -> NCHW once, up front
    nodes.append(helper.make_node("Transpose", ["input"], ["nchw"], perm=[0, 3, 1, 2]))
    cur = "nchw"
    n = 0

    for l in layers:
        cls, c = l["class_name"], l["config"]
        if cls == "InputLayer":
            continue
        if cls == "Dropout":
            continue  # identity at inference
        n += 1
        if cls == "Conv2D":
            wt = weights(c["name"])
            kh, kw = c["kernel_size"]
            sh, sw = c["strides"]
            if c["padding"] == "valid":
                pads = [0, 0, 0, 0]
            else:
                pads = same_pads(h, w, kh, kw, sh, sw)
            wn = init(f"W{n}", wt["kernel"].transpose(3, 2, 0, 1))
            bn = init(f"B{n}", wt["bias"])
            out = f"c{n}"
            nodes.append(helper.make_node(
                "Conv", [cur, wn, bn], [out],
                kernel_shape=[kh, kw], strides=[sh, sw], pads=pads))
            h = (h + pads[0] + pads[2] - kh) // sh + 1
            w = (w + pads[1] + pads[3] - kw) // sw + 1
            cur = out
        elif cls == "MaxPooling2D":
            kh, kw = c["pool_size"]
            sh, sw = c["strides"]
            out = f"p{n}"
            nodes.append(helper.make_node(
                "MaxPool", [cur], [out],
                kernel_shape=[kh, kw], strides=[sh, sw], pads=[0, 0, 0, 0]))
            h = (h - kh) // sh + 1
            w = (w - kw) // sw + 1
            cur = out
        elif cls == "Flatten":
            out = f"f{n}"
            nodes.append(helper.make_node("Flatten", [cur], [out], axis=1))
            cur = out
        elif cls == "Dense":
            wt = weights(c["name"])
            wn = init(f"W{n}", wt["kernel"])  # (in, out): Gemm transB=0 matches
            bn = init(f"B{n}", wt["bias"])
            out = f"d{n}"
            nodes.append(helper.make_node("Gemm", [cur, wn, bn], [out],
                                          alpha=1.0, beta=1.0, transB=0))
            cur = out
        else:
            raise ValueError(f"unsupported layer {cls}")
        act = ACT[c.get("activation")]
        if act:
            aout = f"a{n}"
            if act == "Softmax":
                nodes.append(helper.make_node(act, [cur], [aout], axis=1))
            else:
                nodes.append(helper.make_node(act, [cur], [aout]))
            cur = aout

    nodes.append(helper.make_node("Identity", [cur], ["output"]))
    xout = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])
    graph = helper.make_graph(nodes, "malaria_cnn", [xin], [xout], inits)
    model = helper.make_model(graph, producer_name="h5_to_onnx.py",
                              doc_string=f"Converted from {h5_path}; dropout skipped (inference).")
    model.opset_import[0].version = 17
    model.ir_version = 10  # stay within what onnxruntime-web 1.17 loads
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"wrote {onnx_path}  ({len(nodes)} nodes)")


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/hatch/workspace/diagnostic-demos/sources/Detection-of-Malaria-Using-CNN"
    convert(f"{src}/malaria_cnn.h5", "models/malaria_cnn.onnx", 32)
    convert(f"{src}/my_model.h5", "models/my_model.onnx", 50)
