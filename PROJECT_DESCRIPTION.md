# M3O: Mechanistic Model Merger & Optimizer

A high-performance framework for mechanistic model merging, task vector mathematics, and real-time model collapse heuristics monitoring.

---

## ⚡ What M3O Does

**M3O** transforms open-weight model merging from an opaque guesswork process into an observable, streaming surgical operation built natively for **Gemma 4**.

Instead of loading entire 10 GB checkpoints into RAM all at once, M3O streams Gemma 4’s 140 text-decoder projection tensors incrementally from Safetensors archives. For every layer, it calculates directional task vectors, applies parametric scaling, and monitors mathematical representation drift in real time:

* **Task Vector Arithmetic:** Computes `τ = W_tuned - W_base` on the fly and evaluates the merged state `W_merged = W_base + (α · τ)` layer by layer.
* **Subspace Angular Alignment:** Real-time Cosine Similarity tracking with automated tripwires at the **0.85 Degraded** and **0.65 Critical Collapse** boundaries.
* **Euclidean Drift Magnitude:** Layer-by-layer Frobenius norm displacement (`||ΔW||_F`) to track manifold distortion across network depth (Depth 0 → 34).
* **Dual-Tier Enterprise Telemetry:**
  * **Hot Path (MongoDB Atlas):** Sub-millisecond operational document ingestion for live CLI cockpit alerts and instant crash recovery.
  * **Cold Path (Snowflake):** Micro-batched OLAP warehouse commits enabling historical cross-run SQL analytics and native Streamlit in Snowflake (SiS) governance.
* **Autonomous AI Diagnostician:** Integrates Google’s **`gemma-4-26b-a4b-it`** model via the Gemini API to assess attention subspace preservation and deliver automated hyperparameter tuning recommendations.

---

## 🎯 Key Highlights for Judges

1. **Memory Efficient:** Streams 140 projection tensors at **5–10 layers/sec** with zero memory bloat.
2. **Transparent Observability:** Catches silent representation collapse at the tensor level before expensive inference.
3. **Dual Cloud Pipeline:** Live hot-path streaming to MongoDB Atlas paired with cold-path analytical micro-batches in Snowflake.
4. **End-to-End Gemma 4 Integration:** Powered by Gemma 4 weights and diagnosed by Google's `gemma-4-26b-a4b-it` AI engine.

---

## 🏆 Hackathon Track Alignment

| Track | Implementation in M3O |
| :--- | :--- |
| **Best Open-Source AI Project** | Original, MIT-licensed mechanistic surgery framework performing local tensor algebra on Safetensors weights without black-box APIs. |
| **Best Use of Gemma 4** | Built directly on Gemma 4 open weights (140 projection layers) and uses Google's `gemma-4-26b-a4b-it` MoE model for automated interpretability diagnostics. |
| **Best Use of MongoDB Atlas** | Hot-path real-time telemetry streaming into `m3o_engine.layer_metrics` with sub-millisecond writes and non-blocking server timeouts. |
| **Best Use of Snowflake** | Cold-path analytical warehouse tracking in `M3O_ENGINE.TELEMETRY.LAYER_METRICS` with native Streamlit in Snowflake (SiS) governance. |
| **Best Use of Gemini API** | Serverless invocation of `gemma-4-26b-a4b-it` to analyze layer drift metrics and recommend optimal alpha scaling parameters. |

---

## 🚀 Quick Start Commands

```powershell
# 1. Live Terminal Cockpit (STABLE Pass)
python ui/cli.py --alpha 0.5 --limit 5

# 2. Live Terminal Cockpit (COLLAPSE Stress-Test)
python ui/cli.py --stress-test --limit 5

# 3. Interactive Web Dashboard
python -m streamlit run ui/dashboard.py
```
