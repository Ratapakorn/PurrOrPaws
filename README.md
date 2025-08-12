# PurrOrPaws 🐱🐶
**Cat–vs–Dog image classifier that runs entirely in your browser (no server).**  
Upload a photo and get an instant verdict—your image never leaves your device.

---

## ✨ Demo
- **Live:** `https://<your-username>.github.io/PurrOrPaws/`
- **Repo:** `https://github.com/<your-username>/PurrOrPaws`

*(Update the URLs after you publish.)*

---

## 🚀 Features
- **100% client-side**: ONNX + `onnxruntime-web` (WASM/WebGPU)
- **Fast & private**: No uploads, no backend
- **Tunable threshold**: Balance precision/recall (default **0.410**)
- **Drag & drop** + file picker, sample image included

---

## 🧠 Model
- Base: **MobileNetV2** fine-tuned in Keras (TensorFlow)
- Labels: `cats = 0`, `dogs = 1`
- Export: Keras → **ONNX** via `tf2onnx`
- Browser runtime: **onnxruntime-web**
- Tuned decision threshold: **0.410** (F1 on validation)

---

## 📦 Tech Stack
- Frontend: Vanilla **HTML/CSS/JS**
- Inference: **onnxruntime-web**
- Training: **TensorFlow/Keras**
- Conversion: **tf2onnx**

---

## 🗂 Project Structure
PurrOrPaws/
├─ index.html
├─ app.js
├─ cats_dogs.onnx # exported model
└─ (optional assets)

---

## 🧩 How it works
1. Load `cats_dogs.onnx` in the browser
2. Center-crop & resize to **224×224**
3. Run inference (WASM/WebGPU)
4. Interpret output:
   - **Sigmoid (1 unit)** → `p(dog)`
   - **Softmax (2 units)** → `[p(cat), p(dog)]`
5. Compare `p(dog)` to threshold (default **0.410**) → **cat** or **dog**

---

