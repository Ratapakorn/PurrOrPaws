// Cats vs Dogs — ONNXRuntime Web
// Place cats_dogs.onnx next to this file. Test locally with any static server.
// MIT-licensed snippet; reuse freely.

const MODEL_URL = "./cats_dogs.onnx?v=2"; // ?v=2 busts cache when you update
let THRESHOLD = 0.410;                    // tuned threshold from your training

let session = null;
let inputName = null;
let IMG_W = 224, IMG_H = 224;
let isNCHW = false;      // layout
let outIsSigmoid = true; // 1-unit vs 2-unit

const fileInput = document.getElementById("file");
const sampleBtn = document.getElementById("sample");
const statusEl  = document.getElementById("status");
const preview   = document.getElementById("preview");
const canvas    = document.getElementById("canvas");
const out       = document.getElementById("out");
const tSlider   = document.getElementById("thresh");
const tVal      = document.getElementById("tval");
const drop      = document.getElementById("drop");

const camBtn  = document.getElementById("cam");
const snapBtn = document.getElementById("snap");
const stopBtn = document.getElementById("stop");
const video   = document.getElementById("video");
let mediaStream = null;


// UI init
fileInput.disabled = true; sampleBtn.disabled = true;
tVal && (tVal.textContent = THRESHOLD.toFixed(3));
tSlider?.addEventListener("input", () => {
  THRESHOLD = parseFloat(tSlider.value);
  if (tVal) tVal.textContent = THRESHOLD.toFixed(3);
});

// Drag & drop
["dragover","dragenter"].forEach(ev => drop?.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave","dragend"].forEach(ev => drop?.addEventListener(ev, () => drop.classList.remove("drag")));
drop?.addEventListener("drop", e => {
  e.preventDefault(); drop.classList.remove("drag");
  const f = e.dataTransfer.files?.[0]; if (!f) return;
  const img = new Image(); img.onload = () => runPrediction(img);
  img.src = URL.createObjectURL(f);
});

async function startCamera() {
  try {
    // back camera if available
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    video.srcObject = mediaStream;
    video.style.display = "block";
    camBtn.disabled = true;
    snapBtn.disabled = false;
    stopBtn.disabled = false;
    statusEl.textContent = "Camera ready. Frame your subject and tap Capture.";
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Could not access camera (permission or device issue).";
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  video.srcObject = null;
  video.style.display = "none";
  camBtn.disabled = false;
  snapBtn.disabled = true;
  stopBtn.disabled = true;
  statusEl.textContent = "Camera stopped.";
}

// Draw the current video frame into our preprocessing canvas using center-crop
function drawVideoToCanvasCover() {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, IMG_W, IMG_H);
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return false; // video not ready
  const arVid = vw / vh, arCan = IMG_W / IMG_H;
  let sx, sy, sw, sh;
  if (arVid > arCan) { sh = vh; sw = sh * arCan; sx = (vw - sw)/2; sy = 0; }
  else { sw = vw; sh = sw / arCan; sx = 0; sy = (vh - sh)/2; }
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, IMG_W, IMG_H);
  return true;
}

// Predict directly from the canvas (used after capturing a frame)
async function runPredictionFromCanvas() {
  // Update preview from the canvas snapshot
  preview.src = canvas.toDataURL("image/png");

  const rgba = canvas.getContext("2d").getImageData(0,0,IMG_W,IMG_H).data;
  const N = IMG_W * IMG_H * 3;
  const data = new Float32Array(N);

  if (!isNCHW) {
    for (let i=0, j=0; i<rgba.length; i+=4) { data[j++] = rgba[i]; data[j++] = rgba[i+1]; data[j++] = rgba[i+2]; }
  } else {
    const plane = IMG_W*IMG_H; const R=0, G=plane, B=2*plane;
    for (let y=0, idx=0; y<IMG_H; y++) for (let x=0; x<IMG_W; x++, idx++) {
      const k = (y*IMG_W + x)*4; data[R+idx]=rgba[k]; data[G+idx]=rgba[k+1]; data[B+idx]=rgba[k+2];
    }
  }

  const dims = isNCHW ? [1,3,IMG_H,IMG_W] : [1,IMG_H,IMG_W,3];
  const input = new ort.Tensor("float32", data, dims);

  const t0 = performance.now();
  const results = await session.run({ [inputName]: input });
  const ms = (performance.now() - t0).toFixed(1);

  const outName = Object.keys(results)[0];
  const y = results[outName].data;
  const pDog = outIsSigmoid ? y[0] : y[1];

  const cls = pDog >= THRESHOLD ? "dog" : "cat";
  const conf = pDog >= THRESHOLD ? pDog : 1 - pDog;

  const iconStyle = 'font-size:1.6rem;vertical-align:-2px;margin-right:.4rem;';
  const mutedStyle = 'color:#666;';
  const emoji = cls === "dog" ? "\u{1F436}" : "\u{1F431}";
  out.innerHTML =
    `<span style="${iconStyle}" aria-hidden="true">${emoji}</span>` +
    `<strong>${cls.toUpperCase()}</strong> ` +
    `<span style="${mutedStyle}">(pDog=${pDog.toFixed(3)}, t=${THRESHOLD.toFixed(3)}, ` +
    `confidence=${conf.toFixed(2)}, ${ms} ms)</span>`;
}


async function init() {
  try {
    // Try WebGPU then WASM; fallback to WASM if WebGPU not available
    const providers = ["webgpu","wasm"];
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: providers });

    // ---- Input discovery (robust) ----
    const meta = session.inputMetadata || null;
    const names = meta && Object.keys(meta).length ? Object.keys(meta) :
                  (session.inputNames || []);
    if (!names || !names.length) throw new Error("No input names found.");
    inputName = names[0];

    let dims = null;
    if (meta && meta[inputName] && Array.isArray(meta[inputName].dimensions)) {
      dims = meta[inputName].dimensions;
    }
    if (!dims) dims = [1, 224, 224, 3]; // exported like this

    // normalize dims (replace dynamic with 1)
    dims = dims.map(d => (typeof d === "number" && d > 0 ? d : 1));
    if (dims[3] === 3) { isNCHW = false; IMG_H = dims[1]; IMG_W = dims[2]; }
    else if (dims[1] === 3) { isNCHW = true; IMG_H = dims[2]; IMG_W = dims[3]; }
    else { isNCHW = false; IMG_H = 224; IMG_W = 224; }

    canvas.width = IMG_W; canvas.height = IMG_H;

    // ---- Probe output to decide sigmoid(1) vs softmax(2) ----
    const size = IMG_W * IMG_H * 3;
    const dummy = new ort.Tensor("float32",
      new Float32Array(size),
      isNCHW ? [1,3,IMG_H,IMG_W] : [1,IMG_H,IMG_W,3]
    );
    const feeds = {}; feeds[inputName] = dummy;
    const outMap = await session.run(feeds);
    const outName = Object.keys(outMap)[0];
    const outTensor = outMap[outName];
    outIsSigmoid = (outTensor.dims[outTensor.dims.length - 1] === 1);

    console.log("Model loaded", { inputName, dims, isNCHW, outDims: outTensor.dims, outIsSigmoid });
    statusEl.textContent = `Model ready (${IMG_W}×${IMG_H}, ${isNCHW ? "NCHW" : "NHWC"}, ${outIsSigmoid ? "sigmoid-1" : "softmax-2"})`;
    fileInput.disabled = false; sampleBtn.disabled = false;
  } catch (err) {
    console.error("Model load error:", err);
    statusEl.textContent = "Error loading model. See console.";
  }
}

function drawCover(img) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, IMG_W, IMG_H);
  const arImg = img.width / img.height, arCan = IMG_W / IMG_H;
  let sx, sy, sw, sh;
  if (arImg > arCan) { sh = img.height; sw = sh * arCan; sx = (img.width - sw)/2; sy = 0; }
  else { sw = img.width; sh = sw / arCan; sx = 0; sy = (img.height - sh)/2; }
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, IMG_W, IMG_H);
}

async function runPrediction(img) {
  if (!session) { statusEl.textContent = "Model not ready."; return; }
  preview.src = img.src;
  drawCover(img);

  const rgba = canvas.getContext("2d").getImageData(0,0,IMG_W,IMG_H).data;
  const N = IMG_W * IMG_H * 3;
  const data = new Float32Array(N);

  if (!isNCHW) {
    for (let i=0, j=0; i<rgba.length; i+=4) { data[j++] = rgba[i]; data[j++] = rgba[i+1]; data[j++] = rgba[i+2]; }
  } else {
    const plane = IMG_W*IMG_H; const R=0, G=plane, B=2*plane;
    for (let y=0, idx=0; y<IMG_H; y++) for (let x=0; x<IMG_W; x++, idx++) {
      const k = (y*IMG_W + x)*4; data[R+idx]=rgba[k]; data[G+idx]=rgba[k+1]; data[B+idx]=rgba[k+2];
    }
  }

  const dims = isNCHW ? [1,3,IMG_H,IMG_W] : [1,IMG_H,IMG_W,3];
  const input = new ort.Tensor("float32", data, dims);

  const t0 = performance.now();
  const results = await session.run({ [inputName]: input });
  const ms = (performance.now() - t0).toFixed(1);

  const outName = Object.keys(results)[0];
  const y = results[outName].data;
  const pDog = outIsSigmoid ? y[0] : y[1]; // assume softmax=[p_cat,p_dog]

  const cls = pDog >= THRESHOLD ? "dog" : "cat";
  const conf = pDog >= THRESHOLD ? pDog : 1 - pDog;

  // Inline styles so you don't need a CSS file
  const iconStyle = 'font-size:1.6rem;vertical-align:-2px;margin-right:.4rem;';
  const mutedStyle = 'color:#666;';
  const emoji = cls === "dog" ? "🐶" : "🐱";

  out.innerHTML =
    `<span style="${iconStyle}" aria-hidden="true">${emoji}</span>` +
    `<strong>${cls.toUpperCase()}</strong> ` +
    `<span style="${mutedStyle}">(pDog=${pDog.toFixed(3)}, t=${THRESHOLD.toFixed(3)}, ` +
    `confidence=${conf.toFixed(2)}, ${ms} ms)</span>`;
}

// File + sample handlers
fileInput.addEventListener("change", e => {
  const f = e.target.files?.[0]; if (!f) return;
  const img = new Image(); img.onload = () => runPrediction(img);
  img.src = URL.createObjectURL(f);
});
sampleBtn.addEventListener("click", () => {
  const img = new Image(); img.crossOrigin = "anonymous";
  img.onload = () => runPrediction(img);
  img.src = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg";
});

camBtn?.addEventListener("click", startCamera);
stopBtn?.addEventListener("click", stopCamera);
snapBtn?.addEventListener("click", async () => {
  if (!session) { statusEl.textContent = "Model not ready."; return; }
  if (!mediaStream) { statusEl.textContent = "Camera not started."; return; }
  if (!drawVideoToCanvasCover()) { statusEl.textContent = "Camera warming up… try again."; return; }
  await runPredictionFromCanvas();
});


// Go
init();
