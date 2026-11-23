# Production AI Agents for Generative Media: coursework

My homework and capstone proposal for the 2026 cohort of [Production AI Agents for Generative Media](https://github.com/DataTalksClub/production-ai-agents-for-media-generation) (DataTalks.Club x Nebius). Course material stays in the course repo; this repo holds only my own code, runs and answers.

| Homework | Module | Status | Result |
|---|---|---|---|
| [HW1: GPU Media Worker](hw01-gpu-media-worker/) | 1. Consistent Media Generation on Serverless GPUs | Done | Median `t_gen`: 3.184 s, cost per asset at N=1000: $0.000767, breakeven: 0 assets (never) |
| HW2: Media Agent | 2. Multi-Agent Media Workflows | Not started | |
| HW3: Factory Scale Challenge | 3. Batch Media Generation at Scale | Not started | |
| Capstone proposal | 1 | Not started | |

The capstone itself lives in its own repository.

## Setup

```bash
uv sync --extra dev            # laptop: Endpoint client, cost calculator, tests
uv sync --extra gpu --extra dev  # GPU machine (JupyterLab / DevLab) only
uv run pytest
```

### Repo Structure

```bash
production-ai-agents-coursework/
├── README.md
├── pyproject.toml
├── hw01-gpu-media-worker/
│   ├── README.md
│   ├── generate.py
│   ├── costs.py              # F, c, F/N + c, breakeven
│   ├── results/
│   │   ├── metrics.json      # startup, load, per-asset times, price used, date
│   │   └── assets/           # the 10 outputs (or a sample if large)
│   └── .env.example
├── hw02-media-agent/
├── hw03-factory-scale/
└── capstone-proposal.md
```
