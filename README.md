# m3o: Mechanistic Model Merger & Optimizer

A high-performance framework for mechanistic model merging, task vector mathematics, and real-time model collapse heuristics monitoring.

## Project Structure

```text
m3o/
├── weights/           # Weight tensors storage (.safetensors)
│   ├── model.safetensors         # Tuned Gemma 4 model weights (9.54 GB)
│   └── gemma4_base.safetensors   # Base reference weights (2.21 GB)
├── fixtures/          # Setup helper scripts & test harnesses
│   └── generate_local_base.py    # Base weight generator script
├── engine/            # Task vector math and tensor streaming logic
│   └── merger.py                 # WeightSurgeryEngine & collapse heuristics
├── telemetry/         # Dual-tier telemetry loggers
│   └── logger.py                 # Hot-path MongoDB & Cold-path Snowflake pipeline
├── ui/                # User interfaces
│   ├── cli.py                    # Rich interactive terminal interface
│   └── dashboard.py              # Streamlit interactive dashboard
├── tests/             # Automated test suites
│   ├── test_engine.py            # Engine math & speed benchmarks
│   └── test_telemetry.py         # Telemetry schema & mock pipeline tests
├── requirements.txt   # Core project dependencies
├── .env.example       # Database credentials & configuration template
└── README.md
```

## Setup & Installation

```bash
pip install -r requirements.txt
```

To configure MongoDB Atlas and Snowflake credentials, copy the environment template:
```bash
cp .env.example .env
```
*(If credentials are not configured, M3O automatically runs in fail-safe mock mode).*

## Base Weight Reference Generation

Generate the baseline reference weights `weights/gemma4_base.safetensors` from `weights/model.safetensors`:

```bash
python fixtures/generate_local_base.py
```

## Running the Rich CLI

Execute the streaming weight surgery engine with real-time terminal progress and status badges:

```bash
# Run with default lambda (0.5) in mock telemetry mode
python ui/cli.py --mock-telemetry

# Run with customized lambda and limited layers for quick testing
python ui/cli.py --lambda-val 1.2 --limit 20 --mock-telemetry

# Save merged model weights to a file
python ui/cli.py --lambda-val 0.7 --output-path ./weights/merged_model.safetensors
```

## Launching the Streamlit Dashboard

Launch the interactive web dashboard for real-time drift curves, $\lambda$ tuning, and single-layer inspection:

```bash
python -m streamlit run ui/dashboard.py
```

## Running the Automated Test Suite

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run engine streaming speed benchmark
python tests/test_engine.py
```
