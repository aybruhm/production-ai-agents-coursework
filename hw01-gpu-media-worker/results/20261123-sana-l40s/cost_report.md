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
