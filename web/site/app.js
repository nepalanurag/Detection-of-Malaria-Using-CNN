// Malaria cell classifier demo.
// Runs the shipped my_model.onnx fully in the browser with ONNX Runtime Web.
// Preprocessing matches the analysis: 50x50 RGB, pixels / 255, NHWC float32.
// Output column 0 is P(infected). Uncertainty comes from 8 deterministic
// geometric views (4 rotations x {original, horizontal flip}).

const MODEL_URL = "my_model.onnx?v=2";
const SIZE = 50;

let session = null;
let sampleResults = null;

const $ = (id) => document.getElementById(id);

async function loadModel() {
  $("verdict").textContent = "Loading model\u2026";
  // Single-threaded: the multi-threaded WASM build needs COOP/COEP headers
  // for SharedArrayBuffer, which a static host does not send.
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
  });
  $("verdict").textContent = "Model ready. Upload a cell image.";
}

function drawToCanvas(img, w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(img, 0, 0, w, h);
  return c;
}

// 8 views matching the analysis TTA: rotations of original and h-flipped image.
function makeViews(baseCanvas) {
  const views = [];
  for (const flip of [false, true]) {
    for (const angle of [0, 90, 180, 270]) {
      const c = document.createElement("canvas");
      c.width = SIZE; c.height = SIZE;
      const ctx = c.getContext("2d");
      ctx.translate(SIZE / 2, SIZE / 2);
      ctx.rotate((angle * Math.PI) / 180);
      if (flip) ctx.scale(-1, 1);
      ctx.drawImage(baseCanvas, -SIZE / 2, -SIZE / 2, SIZE, SIZE);
      views.push(c);
    }
  }
  return views;
}

function canvasToTensor(canvas) {
  const data = canvas.getContext("2d").getImageData(0, 0, SIZE, SIZE).data;
  const out = new Float32Array(SIZE * SIZE * 3);
  for (let i = 0; i < SIZE * SIZE; i++) {
    out[i * 3] = data[i * 4] / 255;
    out[i * 3 + 1] = data[i * 4 + 1] / 255;
    out[i * 3 + 2] = data[i * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", out, [1, SIZE, SIZE, 3]);
}

async function predictImage(img, trueLabel) {
  if (!session) await loadModel();
  const base = drawToCanvas(img, SIZE, SIZE);
  const views = makeViews(base);
  const probs = [];
  for (const v of views) {
    const feeds = {};
    feeds[session.inputNames[0]] = canvasToTensor(v);
    const out = await session.run(feeds);
    probs.push(out[session.outputNames[0]].data[0]); // column 0 = P(infected)
  }
  const mean = probs.reduce((a, b) => a + b, 0) / probs.length;
  const sd = Math.sqrt(probs.reduce((a, b) => a + (b - mean) ** 2, 0) / (probs.length - 1));
  const lo = Math.max(0, mean - 1.96 * sd);
  const hi = Math.min(1, mean + 1.96 * sd);
  showResult(img, probs, mean, sd, lo, hi, trueLabel);
}

function showResult(img, probs, mean, sd, lo, hi, trueLabel) {
  $("result").hidden = false;
  const prev = $("preview").getContext("2d");
  prev.imageSmoothingEnabled = false;
  prev.clearRect(0, 0, 128, 128);
  prev.drawImage(img, 0, 0, 128, 128);
  const infected = mean >= 0.5;
  $("verdict").textContent =
    (infected ? "Infected" : "Uninfected") +
    "  \u00b7  P(infected) = " + mean.toFixed(3);
  $("verdict").className = "verdict " + (infected ? "infected" : "uninfected");
  $("interval").textContent =
    "95% interval [" + lo.toFixed(3) + ", " + hi.toFixed(3) + "]" +
    "  (SD " + sd.toFixed(3) + " over 8 views)";
  if (trueLabel) {
    const el = $("truename");
    el.hidden = false;
    el.textContent = "True label of this sample cell: " + trueLabel +
      ". Precomputed locally with the same model; your browser recomputes it live above.";
  } else {
    $("truename").hidden = true;
  }
  const tb = $("viewtable").querySelector("tbody");
  tb.innerHTML = "";
  const names = ["0\u00b0", "90\u00b0", "180\u00b0", "270\u00b0",
                 "flip+0\u00b0", "flip+90\u00b0", "flip+180\u00b0", "flip+270\u00b0"];
  probs.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td>" + names[i] + "</td><td>" + p.toFixed(4) + "</td>";
    tb.appendChild(tr);
  });
  $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(img.src);
    predictImage(img, null).catch((err) => {
      $("verdict").textContent = "Prediction failed: " + (err && err.message ? err.message : err);
    });
  };
  img.src = URL.createObjectURL(file);
}

async function loadSamples() {
  const res = await fetch("samples/results.json");
  sampleResults = await res.json();
  const row = $("samplebuttons");
  sampleResults.samples.forEach((s) => {
    const b = document.createElement("button");
    b.textContent = s.file.replace(".png", "") + " (" + s.true_label + ")";
    b.onclick = () => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => predictImage(img, s.true_label).catch((err) => {
        $("verdict").textContent = "Prediction failed: " + (err && err.message ? err.message : err);
      });
      img.onerror = () => {
        $("verdict").textContent = "Could not load sample image: " + s.file;
      };
      img.src = "samples/" + s.file;
    };
    row.appendChild(b);
  });
}

function init() {
  const dz = $("dropzone");
  const fi = $("fileinput");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") fi.click();
  });
  fi.addEventListener("change", () => handleFile(fi.files[0]));
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
  loadModel().then(loadSamples).catch((err) => {
    $("verdict").textContent = "Could not load the model: " + err.message;
  });
}

init();
