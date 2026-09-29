# Snapdragon Deployment Strategy

> Everything in sections 2–5 is **planned and unverified**. Nothing here was run on Snapdragon hardware, and no latency, power, or speed-up numbers are claimed. The Qualcomm workflow below is written from general knowledge and should be re-checked against current Qualcomm AI Hub / ONNX Runtime documentation before you present it.

## 1. Local Intel implementation (actually built)
scikit-learn model (chosen by validation F1 among LR / RF / GB) + optional ONNX export, run on ONNX Runtime `CPUExecutionProvider`, behind a local FastAPI server. Benchmarks: `reports/local_benchmark.json` (Intel i5-13420H only).

## 2. Snapdragon target architecture
Application → ONNX Runtime (Windows on ARM64 build) → ONNX model → Snapdragon CPU (baseline) / NPU via the QNN execution provider where operators are supported.

Key technical point: tree ensembles (Random Forest / Gradient Boosting) export to ONNX-ML operators (`TreeEnsembleClassifier`), which are generally **CPU-oriented** and not expected to map to the NPU. Logistic Regression or a small MLP export to standard tensor operators and are the realistic candidates for NPU/Qualcomm AI Hub compilation. Because this model is tiny, CPU inference is likely sufficient; NPU offload is not assumed to help.

## 3. Required Snapdragon testing (not done)
1. Run the unchanged ONNX model with ONNX Runtime CPU on a Snapdragon X HP PC; compare probabilities to the Intel results.
2. If using an LR/MLP variant: submit the ONNX model to Qualcomm AI Hub for compile/profile on a target device, then try the QNN execution provider.
3. Measure load time, per-URL latency (median/p95), and memory on-device with `ml/benchmark.py`; measure power with OS/vendor tooling.
4. Verify the ARM64 Python/ONNX Runtime wheels (or a native ARM64 packaging) install cleanly.

## 4. Expected deployment workflow
Train (`ml.train`) → export (`ml.export_onnx`) → validate parity → [optional] compile via AI Hub → package backend + model as an ARM64 Windows app/service → extension talks to `127.0.0.1`.

## 5. Limitations
No Snapdragon measurements exist; NPU suitability depends on the model family; static input shape (1×30 float) is required for most accelerator toolchains; QNN quantization would need re-validation of accuracy.
