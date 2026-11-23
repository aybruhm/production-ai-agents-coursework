# Homework 1: GPU Media Worker

Generate 10 media assets on a serverless GPU in **one run** (one startup, one model load, ten generations), measure the run, and turn the times into cost per asset and a self-host vs per-call API breakeven, using the method from [lesson 05](https://github.com/DataTalksClub/production-ai-agents-for-media-generation/blob/main/01-consistent-media-generation-on-serverless-gpus/05-measuring-costs.md).

Brief: [homework.md](https://github.com/DataTalksClub/production-ai-agents-for-media-generation/blob/main/cohorts/2026/homework/01-consistent-media-generation-on-serverless-gpus/homework.md) · Submit: [course platform](https://courses.datatalks.club/production-ai-agents-generative-media/homework/module1) · Due Mon 30 Nov 2026, 23:59

## The idea

I made a set of ten watercolor storybook illustrations of small winter scenes, such as a red sled leaning against a pine tree or a snow globe with a tiny village inside. Every prompt shares the same template, with soft winter light, warm colors and no people or text, so that the ten images read as one series. I chose this theme because this module is about consistent media generation, and a seasonal set like this only works if the style holds from one image to the next.

## What's here

| File | Role |
|---|---|
| `run.toml` | Everything that defines the run: backend, GPU, model settings, prompt template, the 10 subjects. Copied into `metrics.json` |
| `pricing.toml` | Every price used, with source and read date: GPU $/hour, and the per-call API to compare against |
| `generate.py` | Waits until the model answers, generates each asset once, times each one, writes `assets/` and `metrics.json` |
| `backends.py` | `endpoint`: laptop to Nebius Endpoint (OpenAI-style images API). `local`: Sana in-process on a GPU (JupyterLab / DevLab) |
| `costs.py` | `metrics.json` + `pricing.toml` to `c`, `F`, `F/N + c` at N = 1, 10, 100, 1000, and breakeven N. Writes `cost_report.{json,md}` |
| `tests/` | Cost maths checked against lesson 05's published numbers; generator checked end to end against a fake Endpoint |
| `results/<run_id>/` | The run you submit: assets, `metrics.json`, `cost_report.md`. Scratch runs go to `runs/` (git-ignored) |

## How a number is measured

```
 Create ──► RUNNING ──► model answers ──► asset 1 … asset 10 ──► Stop
 │◄──────── fixed time per run ────────►│◄── t_gen × 10 ──►│◄ tail ►│
            F = price/s × fixed                c = price/s × median t_gen
```

| Quantity | Endpoint backend | Local backend |
|---|---|---|
| Fixed time | `--created-at` to the first `200` from `/v1/models` (provisioning, image pull, weight download and load in one number; the client can't split them). Without `--created-at` it is only a lower bound, and the report says so | `--startup-s` (read off the console) + measured `load_s` |
| Idle | Gap between ready and the first request, plus the tail until you click Stop (`costs.py --stopped-at`). The Endpoint bills while `RUNNING`, so idle is cost | A DevLab bills while it runs, so time spent reading and typing counts too |
| `t_gen` per asset | Wall time per request, network included (the lesson does the same) | GPU time with `torch.cuda.synchronize()` |
| Reported `t_gen` | Median of the 10, so the first-call warm-up doesn't inflate it | Same |

## Run it (Endpoint path, default)

```bash
uv sync --extra dev                 # from the repo root
cd hw01-gpu-media-worker
```

1. Edit `run.toml`: your 10 subjects, the model settings, the GPU you deploy. Keep `billing` in step with `pricing.toml`.
2. Open the template in the console ([Sana](https://console.nebius.com/serverless/endpoint/create?template=sana)), generate and copy the token, then click **Create** and immediately record the time:
   ```bash
   CREATED=$(date -u +%FT%TZ)
   ```
3. Start the run straight away. It asks for the URL and token once and saves them to a git-ignored `.env`:
   ```bash
   uv run python generate.py --created-at "$CREATED" --out results/$(date -u +%Y%m%d)-sana-l40s
   ```
   `502` responses while the model loads are expected; it polls every 2 s for up to 15 minutes. Stop it with **Ctrl-C** if needed: the record is kept with `status: failed`.
4. **Stop the Endpoint** as soon as it prints the reminder, and record the time: `STOPPED=$(date -u +%FT%TZ)`.
5. Re-read today's prices into `pricing.toml` (GPU: [Nebius pricing](https://nebius.com/prices); API: the model's page), then:
   ```bash
   uv run python costs.py results/<run_id>/metrics.json --stopped-at "$STOPPED"
   ```
6. Paste `cost_report.md` into [Results](#results).

**Local path (JupyterLab / DevLab):** `uv sync --extra gpu --extra dev`, then `uv run python generate.py --backend local --startup-s <seconds from Create to a usable kernel>`. Use the regular DevLab price in `pricing.toml`. Delete the DevLab afterwards: a stopped one still bills its disk.

## Know before you run: Sana vs fal is close

fal charges $0.001 per megapixel for Sana, so at 512x512 one image costs $0.000262. On an L40S at $0.7492/h a GPU-second costs $0.000208, so **self-hosting can only break even if your median `t_gen` is below 1.26 s** (0.000262 / 0.000208). Above that, `c ≥ p` and the breakeven is 0 (never). The lesson measured about 1.2 s per 512x512 image through an Endpoint on an H100. An L40S may come out slower.

At 1024x1024 the API price quadruples ($0.00105) and the threshold moves to 5.04 s, against the 4.42 s that lesson 01 measured on an L40S. Either result is a valid answer; it just has to be honest and explained. Also check how fal rounds megapixels before relying on per-pixel pricing.

## Results

Run folder: [`results/20261123-sana-l40s/`](results/20261123-sana-l40s/) (assets, `metrics.json`, `cost_report.md`).

| Quantity | Value |
|---|---|
| Model | `Efficient-Large-Model/Sana_1600M_1024px_diffusers` |
| How run | endpoint |
| GPU | L40S (gpu-l40s-a, 1gpu-8vcpu-32gb, eu-north1) |
| Price used | $0.7492/h, preemptible, read 2026-11-11 |
| Assets | 10 |
| Fixed time per run | 503.2 s |
| Median generation time | 3.184 s |
| Marginal cost per asset, c | $0.000663 |
| Fixed cost per run, F | $0.1047 |
| Cost per asset at N=1 | $0.105384 |
| Cost per asset at N=10 | $0.011135 |
| Cost per asset at N=100 | $0.001710 |
| Cost per asset at N=1000 | $0.000767 |
| This run, projected | $0.1122 |
| API compared | fal `fal-ai/sana`, $0.000262 per asset (0.001 per megapixel, read 2026-11-23), https://fal.ai/models/fal-ai/sana |
| Breakeven N | 0 (never: the API is cheaper per asset) |

- fixed time basis: endpoint created_at -> /v1/models ready
- includes 496.7 s idle after the last asset until Stop

The median generation time came out at 3.184 s per 512x512 image, which is well above the 1.26 s threshold worked out in the section above. Because of this, the marginal cost per asset on the L40S ($0.000663) is about 2.5 times the fal price ($0.000262), so self-hosting never breaks even at this size and the breakeven N is 0. The first request took 6.99 s because of warm-up, which is why I report the median rather than the mean.

The fixed time of 503.2 s needs some explanation. Only 6.5 s of it is the time from Create to the first answer from `/v1/models`, which is a lot shorter than a cold start would normally take, so I believe the Endpoint was already up by the time I recorded the Create time. The other 496.7 s is idle time between the last asset and clicking Stop, which the Endpoint still bills for. This means F here is mostly idle time, and a run with a true cold start would have a higher F. It would not change the breakeven, since c is already above the API price.
