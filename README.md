# Agentic Root Cause Analysis Copilot

A production-style GenAI reference implementation for diagnosing industrial equipment faults using predictive maintenance data, engineering knowledge, and agentic verification loops.

## Overview

This repository demonstrates (Agentic Harness) a stateful, audited Root Cause Analysis (RCA) workflow built on:

- LangGraph multi-agent control loops
- LangChain tools for data analytics, retrieval, and reporting
- Local-first LLM support with Ollama plus optional OpenAI/Anthropic providers
- Structured evidence grounding, verification, and refinement
- Safety, permissions, and prompt-injection filtering
- Session persistence and Markdown report export



## Key Features

- Multi-agent RCA graph with:
  - data interpretation
  - failure classification
  - knowledge retrieval (RAG)
  - root cause reasoning
  - verification and refinement
  - action planning and report writing
- Runtime skills registry loaded from `skills.md`
- Analytics tools for sensor trends, anomaly detection, and fault lookup
- Evidence viewer and evaluation dashboard for groundedness and risk
- SQLite session store for replayable RCA runs
- Exportable Markdown reports in `exports/`

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full architecture overview and [ROADMAP.md](ROADMAP.md) for planned enhancements.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/download_ai4i.py
streamlit run streamlit_app.py
```

### Optional Ollama Setup

If you want local LLM execution with Ollama:

```bash
ollama pull llama3.1
ollama serve
```

The app can still run with deterministic fallback reasoning if no LLM provider is available.

## Dataset

The default data loader downloads the AI4I 2020 Predictive Maintenance Dataset from UCI. The app also supports custom CSV uploads via the Streamlit UI.

Expected AI4I columns include:

- `UDI`
- `Product ID`
- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Machine failure`
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF`

## Usage

1. Run the Streamlit app.
2. Load `data/raw/ai4i2020.csv` or upload another dataset.
3. Select a failure event or record to analyze.
4. Ask a focused RCA question such as:
   - `Why did this equipment likely fail?`
   - `What should maintenance inspect first?`
5. Review the generated report, evidence panels, and evaluation metrics.

## Project Layout

- `app/` — application logic, agents, UI components, and core services
- `data/` — dataset loaders, manuals, fault references, and exported reports
- `scripts/` — helper scripts such as dataset downloaders
- `skills.md` — runtime skill registry definitions
- `exports/` — generated RCA reports and Markdown exports
- `streamlit_app.py` — Streamlit entrypoint

## Notes

- This project is a reference implementation for engineering decision support.
- It is not a substitute for OEM guidance, inspection procedures, or lockout/tagout practices.
- Use the safety checks and evidence viewer to validate model outputs before acting on them.

## License

No license file is included in this repository. Add one if you plan to distribute or publish this project.

