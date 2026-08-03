# MiniMind-Align

MiniMind-Align is an auditable alignment research snapshot covering error-driven SFT, DPO, SimPO, corrected-GRPO, validator replay, precision contracts, and checkpoint diagnostics.

## Main result

Error-driven SFT, DPO, and SimPO improved independent quality validation. Corrected-GRPO produced no additional validation gain over its SFT source model, so the default model was intentionally not replaced.

| Method | Full validation | Release slice | Status |
|---|---:|---:|---|
| Baseline | 48/160 | 13/32 | Reference |
| Error-driven SFT | 79/160 | 19/32 | Best diagnostic candidate |
| DPO | 79/160 | 19/32 | Diagnostic candidate |
| SimPO | 76/160 | 19/32 | Diagnostic candidate |
| Corrected-GRPO | — | 19/32 | No incremental gain |

Run the public, CPU-only checks with:

~~~bash
bash scripts/reproduce_public_audit.sh
bash scripts/reproduce_public_smoke.sh
~~~

Weights, raw training data, full generations, and cloud credentials are intentionally excluded. The base MiniMind implementation is derived from [jingyaogong/minimind](https://github.com/jingyaogong/minimind) at commit `307fd76` under Apache-2.0; this README describes the MiniMind-Align research contribution. See [docs/upstream.md](docs/upstream.md), [docs/model_card.md](docs/model_card.md), and [results/public/limitations.md](results/public/limitations.md).
