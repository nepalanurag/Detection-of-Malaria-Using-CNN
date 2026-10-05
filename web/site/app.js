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

/* Malaria around the world: WHO country data, hand-drawn canvas chart.
   data.csv columns: index, Country, Year, No. of cases, No. of deaths, WHO Region. */
(function () {
  var chart = document.getElementById("malchart");
  if (!chart) return;
  var countrySel = document.getElementById("country");
  var metricSel = document.getElementById("metric");
  var yearSlider = document.getElementById("yearslider");
  var top5year = document.getElementById("top5year");
  var top5 = document.getElementById("top5");
  var rows = [];

  function fmtAxis(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
    return String(Math.round(n));
  }
  function fmtFull(n) { return Math.round(n).toLocaleString("en-US"); }

  function parseCSV(text) {
    var out = [];
    var lines = text.trim().split(/\r?\n/);
    for (var i = 1; i < lines.length; i++) {
      var p = lines[i].split(",");
      if (p.length < 6) continue;
      out.push({
        country: p[1],
        year: parseInt(p[2], 10),
        cases: parseFloat(p[3]) || 0,
        deaths: parseFloat(p[4]) || 0
      });
    }
    return out;
  }

  function draw() {
    var country = countrySel.value;
    var metric = metricSel.value;
    var pts = rows.filter(function (r) { return r.country === country; })
                  .sort(function (a, b) { return a.year - b.year; });
    var ctx = chart.getContext("2d");
    var W = chart.width, H = chart.height;
    ctx.clearRect(0, 0, W, H);
    if (!pts.length) return;
    var padL = 52, padR = 12, padT = 12, padB = 28;
    var maxV = Math.max.apply(null, pts.map(function (p) { return p[metric]; }).concat([1]));
    var minY = pts[0].year, maxY = pts[pts.length - 1].year;
    function X(y) { return padL + ((y - minY) / Math.max(1, maxY - minY)) * (W - padL - padR); }
    function Y(v) { return H - padB - (v / maxV) * (H - padT - padB); }
    ctx.font = "11px sans-serif";
    ctx.lineWidth = 1;
    var g, v, yy;
    for (g = 0; g <= 4; g++) {
      v = (maxV * g) / 4; yy = Y(v);
      ctx.strokeStyle = "#e5e5e5";
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(W - padR, yy); ctx.stroke();
      ctx.fillStyle = "#666";
      ctx.fillText(fmtAxis(v), 4, yy + 4);
    }
    ctx.fillStyle = "#666";
    pts.forEach(function (p) {
      if ((p.year - minY) % 3 === 0 || p.year === maxY)
        ctx.fillText(String(p.year), X(p.year) - 12, H - 10);
    });
    ctx.strokeStyle = "#8c2b2b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(function (p, i) {
      var x = X(p.year), y = Y(p[metric]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#8c2b2b";
    pts.forEach(function (p) {
      ctx.beginPath(); ctx.arc(X(p.year), Y(p[metric]), 3, 0, 6.3); ctx.fill();
    });
  }

  function renderTop5() {
    var yr = parseInt(yearSlider.value, 10);
    top5year.textContent = yr;
    var list = rows.filter(function (r) { return r.year === yr; })
                   .sort(function (a, b) { return b.cases - a.cases; })
                   .slice(0, 5);
    top5.innerHTML = "";
    list.forEach(function (r) {
      var li = document.createElement("li");
      li.textContent = r.country + ": " + fmtFull(r.cases) + " cases";
      top5.appendChild(li);
    });
  }

  fetch("data.csv").then(function (r) { return r.text(); }).then(function (t) {
    rows = parseCSV(t);
    var countries = [];
    rows.forEach(function (r) { if (countries.indexOf(r.country) < 0) countries.push(r.country); });
    countries.sort();
    countries.forEach(function (c) {
      var o = document.createElement("option");
      o.value = c; o.textContent = c;
      countrySel.appendChild(o);
    });
    // default: country with the most cases in the latest year
    var maxYr = Math.max.apply(null, rows.map(function (r) { return r.year; }));
    var top = rows.filter(function (r) { return r.year === maxYr; })
                  .sort(function (a, b) { return b.cases - a.cases; })[0];
    if (top) countrySel.value = top.country;
    countrySel.addEventListener("change", draw);
    metricSel.addEventListener("change", draw);
    yearSlider.addEventListener("input", renderTop5);
    draw();
    renderTop5();
  }).catch(function () {
    chart.getContext("2d").fillText("Could not load data.csv", 20, 40);
  });
})();

/* Cloud inference: POST a cell image to /api/predict and show the result.
   Same model and same 8-view TTA as the in-browser flow, run server-side. */
(function () {
  var fi = document.getElementById("cloudfile");
  var btn = document.getElementById("cloudbtn");
  var box = document.getElementById("cloudresult");
  var verdict = document.getElementById("cloudverdict");
  var interval = document.getElementById("cloudinterval");
  var err = document.getElementById("clouderror");
  if (!fi || !btn) return;

  function showError(msg) {
    box.hidden = false;
    verdict.textContent = "Cloud classification failed.";
    verdict.className = "verdict";
    interval.textContent = "";
    err.hidden = false;
    err.textContent = msg;
  }

  btn.addEventListener("click", function () {
    var f = fi.files[0];
    if (!f || f.type.indexOf("image/") !== 0) {
      showError("Choose a JPG or PNG image first.");
      return;
    }
    verdict.textContent = "Classifying in the cloud\u2026";
    verdict.className = "verdict";
    interval.textContent = "";
    err.hidden = true;
    box.hidden = false;
    var reader = new FileReader();
    reader.onload = function () {
      fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: reader.result })
      }).then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, data: data };
        });
      }).then(function (r) {
        if (!r.ok) {
          showError(r.data && r.data.error ? r.data.error : "The server returned an error.");
          return;
        }
        var infected = r.data.label === "infected";
        verdict.textContent =
          (infected ? "Infected" : "Uninfected") +
          "  \u00b7  P(infected) = " + Number(r.data.probability).toFixed(3) +
          "  \u00b7  classified in the cloud";
        verdict.className = "verdict " + (infected ? "infected" : "uninfected");
        interval.textContent =
          "95% interval [" + Number(r.data.ci_low).toFixed(3) + ", " +
          Number(r.data.ci_high).toFixed(3) + "]  (" +
          r.data.n_augmentations + " views, " + r.data.model + ")";
      }).catch(function () {
        showError("Could not reach the cloud API. Check your connection and try again.");
      });
    };
    reader.readAsDataURL(f);
  });
})();
