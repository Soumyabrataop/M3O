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

## 🏗️ Architecture

```text
                  ┌─────────────────────────────────────────┐
                  │       Streaming Surgery Engine          │
                  │   (5-10 Gemma 4 projection layers/sec)  │
                  └────────────────────┬────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │       HOT PATH        │             │       COLD PATH       │
        │     MongoDB Atlas     │             │  Snowflake Warehouse  │
        ├───────────────────────┤             ├───────────────────────┤
        │ • Sub-millisecond write│             │ • Micro-batched flush │
        │ • Flexible JSON doc   │             │ • Columnar OLAP SQL   │
        │ • Real-time alerts    │             │ • Long-term governance│
        │ • Sub-second cockpit  │             │ • Native SiS Dashboard│
        └───────────────────────┘             └───────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │      Gemma 4 AI Diagnostician           │
                  │  (gemma-4-26b-a4b-it via Gemini API)    │
                  └─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
M3O/
├── weights/                      # Safetensors weight checkpoints (ignored from git)
│   ├── model.safetensors         # Tuned Gemma 4 model weights (9.54 GB)
│   └── gemma4_base.safetensors   # Base reference weights (2.21 GB)
├── fixtures/                     # Test harnesses & fixture generators
│   └── generate_local_base.py    # Generates local base checkpoint
├── engine/                       # Task vector math and streaming engine
│   └── merger.py                 # WeightSurgeryEngine & collapse heuristics
├── telemetry/                    # Dual-tier telemetry & AI diagnostician
│   ├── logger.py                 # Hot-path MongoDB & Cold-path Snowflake pipeline
│   └── gemini_diagnostics.py     # Live gemma-4-26b-a4b-it mechanistic analyzer
├── ui/                           # User interfaces
│   ├── cli.py                    # Live terminal cockpit with Rich formatting
│   ├── dashboard.py              # Interactive Streamlit dashboard
│   └── snowflake_app.py          # Native Streamlit in Snowflake (SiS) app
├── tests/                        # Automated unit & benchmark tests
│   ├── test_engine.py            # Engine math & speed benchmarks
│   └── test_telemetry.py         # Telemetry schema & mock pipeline tests
├── requirements.txt              # Core project dependencies
├── .env.example                  # Environment configuration template
├── PROJECT_DESCRIPTION.md        # Submission writeup & judge highlights
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
cp .env.example .env
```

### 2. Prepare Weights

Place your safetensors weight files in the `weights/` directory:
- `weights/model.safetensors` - Your tuned model weights
- `weights/gemma4_base.safetensors` - Base reference weights

You can generate synthetic base weights from your tuned weights:
```bash
python fixtures/generate_local_base.py
```

### 3. Live Terminal Surgery Cockpit

```powershell
# Basic usage with default alpha (0.7)
python ui/cli.py

# Run with custom alpha scaling
python ui/cli.py --alpha 0.5 --limit 10

# Demonstrate representation collapse detection
python ui/cli.py --stress-test --limit 5

# Run simulated collapse test (no weight files needed)
python ui/cli.py --stress-simulate --limit 20

# Save merged weights to file
python ui/cli.py --alpha 0.5 --output ./weights/merged.safetensors

# Use mock telemetry (no database connections)
python ui/cli.py --mock --limit 5
```

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--alpha` | 0.7 | Task vector scaling coefficient |
| `--base` | `./weights/gemma4_base.safetensors` | Path to base reference weights |
| `--tuned` | `./weights/model.safetensors` | Path to tuned weights |
| `--output` | None | Output path to save merged weights |
| `--limit` | None | Limit number of layers to process |
| `--stress-test` | False | Run comprehensive stress test (tests α from -1.0 to 10.0 to find collapse boundaries) |
| `--stress-simulate` | False | Run simulated collapse test (no weight files needed, demonstrates DEGRADED/CRITICAL behavior) |
| `--mock` | False | Force mock telemetry mode (no DB connections) |
| `--mongo` | None | MongoDB Atlas connection URI (overrides env) |
| `--gemini-key` | None | Gemini API key for AI diagnostics (overrides env) |

### 4. Interactive Web Dashboard

```powershell
python -m streamlit run ui/dashboard.py
```

### 5. Native Streamlit in Snowflake (SiS)

Deploy `ui/snowflake_app.py` directly inside Snowflake Snowsight (Projects → Streamlit) pointing to `M3O_ENGINE.TELEMETRY.LAYER_METRICS`.

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

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
