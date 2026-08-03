# 复现说明

在 `.`、Python 3.10 `.venv` 和可用 L4 上执行。所有长期任务放入 tmux；运行前确认磁盘至少剩余 30GB。

```bash
bash scripts/reproduce_all.sh --dry-run
bash scripts/reproduce_all.sh --stage data --smoke
bash scripts/reproduce_all.sh --stage preferences --smoke
bash scripts/reproduce_all.sh --stage simpo --smoke
bash scripts/reproduce_all.sh --stage reward --smoke
bash scripts/reproduce_all.sh --stage rl --smoke
bash scripts/reproduce_all.sh --stage evaluation --smoke
bash scripts/run_gemini_pair_extension.sh dpo
bash scripts/run_gemini_pair_extension.sh simpo
bash scripts/run_gemini_pair_extension.sh simpo_full
GEMINI_SEED=43 bash scripts/run_gemini_pair_extension.sh simpo_full seed43
```

`--full` 只在确认资源与输出目录后使用：

```bash
bash scripts/reproduce_all.sh --stage simpo --full
bash scripts/reproduce_all.sh --stage reward --full
bash scripts/reproduce_all.sh --stage rl --full
```

每个脚本对应一个 Sprint：`reproduce_data_v2.sh`、`reproduce_sft_v2.sh`、`reproduce_preferences_v2.sh`、`reproduce_simpo.sh`、`reproduce_reward_model.sh`、`reproduce_rl.sh`、`reproduce_evaluation.sh`。脚本只使用独立 `results/experiments/reproduce_*` 路径；正式已有权重不应作为输出目标。

验证：

```bash
.venv/bin/python -m py_compile align/reward_model.py align/rl_rules.py trainer/train_reward.py trainer/train_grpo_lite.py
PYTHONPATH=. .venv/bin/python -m unittest tests.test_reward_model tests.test_simpo tests.test_rl_rules
.venv/bin/python evaluation/build_unified_report.py
```

冻结评测设置见 `results/experiments/unified_sft_v2_20260731/unified_metrics.json`；不要更换测试 prompt、seed 或 decoding 后直接比较数值。

Gemini 扩展脚本只用于已有冻结生成结果的匿名 A/B 评审；它会创建独立 experiment id，并支持在已有 `judge.jsonl` 上断点续跑。`GEMINI_SEED` 可改变匿名顺序复核的随机种子；本项目的 SimPO full seed=43 复核结果保存在 `results/experiments/gemini_align_vs_simpo_v1_full_seed43_20260731/`。需要确认 Google Cloud 礼金、Vertex 权限和实际成本口径后再启动。

C003 on-policy Gemini 全量排名使用已生成的 pair，不改动训练权重，也不把 judgment 回流训练：

~~~bash
.venv/bin/python evaluation/build_gemini_rank_inputs.py \
  --pairs results/experiments/on_policy_c001_train_20260731/pairs.jsonl \
  --baseline-output results/inputs/gemini_c003_train_full_baseline_20260801.jsonl \
  --candidate-output results/inputs/gemini_c003_train_full_candidate_20260801.jsonl

bash scripts/run_experiment.sh \
  --experiment-id gemini_on_policy_c003_train_full_20260801 \
  --task-id MM-C003 \
  -- .venv-teacher/bin/python evaluation/judge_generation_gemini.py \
  --baseline results/inputs/gemini_c003_train_full_baseline_20260801.jsonl \
  --candidate results/inputs/gemini_c003_train_full_candidate_20260801.jsonl \
  --baseline_name hard_rejected --candidate_name validator_chosen \
  --output results/experiments/gemini_on_policy_c003_train_full_20260801/judge.jsonl \
  --summary results/experiments/gemini_on_policy_c003_train_full_20260801/summary.json \
  --review results/experiments/gemini_on_policy_c003_train_full_20260801/review.md \
  --project gen-lang-client-0131552860 --location global --model gemini-3.6-flash \
  --seed 42 --sleep 0.4 --max_retries 5

bash scripts/run_experiment.sh \
  --experiment-id gemini_on_policy_c003_validation_full_20260801 \
  --task-id MM-C003 \
  -- .venv-teacher/bin/python evaluation/judge_generation_gemini.py \
  --baseline results/inputs/on_policy_validation_hard_rejected_20260731.jsonl \
  --candidate results/inputs/on_policy_validation_validator_chosen_20260731.jsonl \
  --baseline_name hard_rejected --candidate_name validator_chosen \
  --output results/experiments/gemini_on_policy_c003_validation_full_20260801/judge.jsonl \
  --summary results/experiments/gemini_on_policy_c003_validation_full_20260801/summary.json \
  --review results/experiments/gemini_on_policy_c003_validation_full_20260801/review.md \
  --project gen-lang-client-0131552860 --location global --model gemini-3.6-flash \
  --seed 42 --sleep 0.4 --max_retries 5
~~~

已完成结果见 docs/experiments/gemini_on_policy_full_ranking.md；正式评测应继续使用独立 experiment id，避免覆盖已有 judgment。

RL extended 对照使用独立输出目录和未参与 RL 训练的冻结测试集：

~~~bash
COMMON=(--manifest results/inputs/on_policy_validation_manifest_32_20260731.jsonl --categories conciseness,format,instruction,reasoning,repetition,safety,termination,uncertainty --max-prompts 32 --from-weight align_sft_v2_pilot --model-dir out --tokenizer-path model --batch-size 1 --accumulation-steps 4 --num-generations 8 --max-seq-len 384 --max-gen-len 96 --max-steps 8 --learning-rate 3e-7 --beta 0.02 --epsilon 0.2 --epsilon-high 5.0 --dtype bfloat16 --device cuda:0 --seed 42)
.venv/bin/python trainer/train_grpo_lite.py ${COMMON[@]} --mode grpo --save-dir results/experiments/grpo_lite_extended_20260801/out --save-weight grpo_extended
.venv/bin/python trainer/train_grpo_lite.py ${COMMON[@]} --mode cispo --save-dir results/experiments/cispo_lite_extended_20260801/out --save-weight cispo_extended
.venv/bin/python evaluation/generate_frozen_test.py --weight grpo_extended --model-name grpo_extended --model-dir results/experiments/grpo_lite_extended_20260801/out --tokenizer-path model --output results/experiments/grpo_extended_frozen_eval_20260801/generation.jsonl
.venv/bin/python evaluation/score_frozen_test.py --generation results/experiments/grpo_extended_frozen_eval_20260801/generation.jsonl --output-dir results/experiments/grpo_extended_frozen_eval_20260801/score
.venv/bin/python evaluation/generate_frozen_test.py --weight cispo_extended --model-name cispo_extended --model-dir results/experiments/cispo_lite_extended_20260801/out --tokenizer-path model --output results/experiments/cispo_extended_frozen_eval_20260801/generation.jsonl
.venv/bin/python evaluation/score_frozen_test.py --generation results/experiments/cispo_extended_frozen_eval_20260801/generation.jsonl --output-dir results/experiments/cispo_extended_frozen_eval_20260801/score
~~~

结果、哈希和限制见 docs/experiments/grpo_cispo_extended_20260801.md。

## MM-E009 / MM-F015 复现

正式 RL 三 seed 串行 wrapper：

```bash
bash scripts/run_rl_suite.sh --dry-run
bash scripts/run_rl_suite.sh --full
```

C-Eval 代表子集（要求固定 revision；脚本默认使用本轮已锁定 revision）：

```bash
bash scripts/run_ceval_subset.sh
```

smoke 使用 `scripts/run_rl_suite.sh --smoke` 和 evaluator 的 `--smoke --subjects high_school_chinese --questions-per-subject 2`。所有正式输出必须使用新的 experiment directory；wrapper 拒绝复用既有目录。C-Eval 的 val/dev 补足、科目 alias、manifest hash 和每模型 predictions 以 `source.json`/`summary.json` 为准。

## MM-E010 / MM-F017 复现

先准备并审计原生 Alignment v2 数据，再运行 checkpoint 回载修正版 wrapper：

```bash
bash scripts/run_rl_data_isolation_reload_fixed.sh --dry-run
bash scripts/run_rl_data_isolation_reload_fixed.sh --smoke
bash scripts/run_rl_data_isolation_reload_fixed.sh --full
```

wrapper 固定使用 `results/inputs/rl_data_isolation_train_128_20260801.jsonl`、`results/inputs/rl_data_isolation_validation_32_20260801.jsonl`，一次只运行一个 GPU 任务，并把六个正式 seed 的日志、checkpoint、validation history、selection 和资源记录写入新的 `results/experiments/rl_data_isolation_reload_fixed_20260801/`。训练器在每个 checkpoint 保存后回载 artifact，再计算 validation、quality gate 和 selection；不要把旧 `rl_data_isolation_20260801` 目录的内存指标与本轮结果合并。

```bash
.venv/bin/python evaluation/audit_rl_reward_hacking.py \
  --experiment-root results/experiments/rl_data_isolation_reload_fixed_20260801 \
  --output-dir results/experiments/rl_data_isolation_reload_fixed_20260801/reward_hacking_audit
```

wrapper 会自动执行 baseline 与每个唯一 selected checkpoint 的 100 条冻结集 validator 评测；本阶段不重跑 C-Eval。审计 warning 只用于诊断，不得绕过 KL、safety/termination 或三 seed 晋级门禁。完成后核对 6 个 run 的 `evaluation_source=reloaded_checkpoint`、JSON/JSONL 无 NaN、exit code、GPU wall time 和服务器状态。

完整 v2 validation 回载诊断（只读、不改权重）可复现为：

```bash
bash scripts/run_rl_validation_coverage.sh --dry-run
bash scripts/run_rl_validation_coverage.sh --full
```

它评估 160 条完整 validation 和预注册 32 条切片；必须同时检查
`results/experiments/rl_validation_coverage_20260801/coverage_summary.json` 中的
`checkpoint_reproducibility`。若继续训练，选点前必须回载保存后的 checkpoint 再计算 validation，不能只使用训练进程内存中的指标。

当前训练器已内置该回载流程；相关 smoke 结果在
`results/experiments/rl_checkpoint_reload_smoke_20260801/`。复现时检查
`validation_history.jsonl` 的 `evaluation_source` 必须为 `reloaded_checkpoint`，并用
`max_steps=0` 对选中 checkpoint 做一次独立回载核对。

## MM-E011 / MM-F018 稳定性诊断复现

稳定性诊断使用独立的 v2 data-isolation manifests，不覆盖旧实验：

```bash
bash scripts/run_rl_stability_diagnostic.sh --dry-run
bash scripts/run_rl_stability_diagnostic.sh --smoke
bash scripts/run_rl_stability_diagnostic.sh --full
```

wrapper 固定串行运行 GRPO/CISPO 的 `control`、`low_lr`、`accum16`，seed=42；每 4 steps validation/checkpoint，最多 20 steps，预算上限 7200 秒，使用 tmux 和 60 秒资源轮询。完整产物位于 `results/experiments/rl_stability_diagnostic_20260801/`。审计可单独运行：

```bash
.venv/bin/python evaluation/audit_rl_stability.py \
  --experiment-root results/experiments/rl_stability_diagnostic_20260801 \
  --output-dir results/experiments/rl_stability_diagnostic_20260801/stability_audit
```

验收时检查 `stability_audit/summary.json` 的 `all_json_finite`、`stability_improved_conditions`、六个 run 的 exit code、`run.log` 中的 commit/environment/GPU/disk/wall time，以及每个 selected checkpoint 的回载结果。本阶段不重跑 C-Eval 或冻结集；稳定性 warning 只作诊断，不改变 KL、quality 或模型晋级门禁。

## MM-F019 KL 尖峰归因复现

该步骤只读取稳定性实验产物，不启动 GPU：

```bash
.venv/bin/python evaluation/audit_rl_spike_sources.py \
  --experiment-root results/experiments/rl_stability_diagnostic_20260801 \
  --output-dir results/experiments/rl_stability_diagnostic_20260801/spike_source_audit_v2
```

审计默认只选择六个正式 `grpo/cispo × control/low_lr/accum16 × seed42` run；`--include-smoke` 才包含 smoke。输出包括 `summary.json`、逐 run `run_reports.jsonl` 和 `report.md`。审计只做 step/sample-level 解释，不改变任何模型门禁；若输出目录非空，脚本拒绝覆盖。

## MM-E012 / MM-F020 micro-batch 尖峰归因复现

训练器和 wrapper 位于 `trainer/train_grpo_lite.py`、`evaluation/audit_rl_spike_sources.py` 与 `scripts/run_rl_microbatch_telemetry.sh`。所有新运行写入独立根目录，并拒绝复用既有 run 目录：

```bash
bash scripts/run_rl_microbatch_telemetry.sh --dry-run
bash scripts/run_rl_microbatch_telemetry.sh --smoke
bash scripts/run_rl_microbatch_telemetry.sh --diagnostic
```

正式诊断使用 Alignment v2 隔离 train/validation manifest、GRPO seed=42、4 steps、8 generations、8 accumulation steps、`max_grad_norm=1.0`、`bfloat16`，并通过 `--interleave-categories` 让 32 个 micro-batch 覆盖八类。wrapper 使用 tmux、60 秒资源轮询和 4 小时硬上限；run.log 保存命令、commit、environment hash、GPU/磁盘、退出码和 wall time。

最终 balanced run 的离线审计可复现为：

```bash
.venv/bin/python evaluation/audit_rl_spike_sources.py \
  --experiment-root results/experiments/rl_microbatch_telemetry_20260801 \
  --output-dir results/experiments/rl_microbatch_telemetry_20260801/audit_balanced_v2 \
  --run-name grpo_control_seed42_diagnostic_balanced \
  --require-microbatch
```

`audit_balanced_v2/` 已由既有 run 生成；若重新运行，需使用新的空输出目录。输出包括 `summary.json`、`run_reports.jsonl`、`report.md`、`microbatch_attribution.json`、`category_summary.json` 和 `prompt_summary.json`。审计只产生诊断 warning，不改变 checkpoint eligibility 或模型晋级门禁。

## MM-E013 / MM-F021 更新尺度稳定性对照复现

入口脚本为 `scripts/run_rl_update_scale_diagnostic.sh`，使用独立根目录，且每个 run 目录拒绝覆盖：

```bash
bash scripts/run_rl_update_scale_diagnostic.sh --dry-run
bash scripts/run_rl_update_scale_diagnostic.sh --smoke
bash scripts/run_rl_update_scale_diagnostic.sh --full
```

formal 条件固定为 GRPO seed=42、8 steps、8 generations、8 accumulation steps、`max_seq_len=384`、`max_gen_len=128`、`eval_every=2`、`checkpoint_every=4`、`bfloat16`、KL threshold `0.005`/patience `2`，只改变：

```text
control:   learning_rate=3e-7, max_grad_norm=1.0
low_lr:    learning_rate=1e-7, max_grad_norm=1.0
clip_half: learning_rate=3e-7, max_grad_norm=0.5
```

训练开启 `--microbatch-gradient-norm` 和 `--interleave-categories`；wrapper 使用 tmux、60 秒资源轮询和 7200 秒硬上限。训练后由 wrapper 运行 stability audit 和三个独立 spike-source audit。当前最终 stability 输出为：

```bash
.venv/bin/python evaluation/audit_rl_stability.py \
  --experiment-root results/experiments/rl_update_scale_diagnostic_20260801/formal \
  --output-dir results/experiments/rl_update_scale_diagnostic_20260801/formal/stability_audit_v2
```

三条件 micro-batch audit 输出分别位于 `formal/spike_audit_control/`、`formal/spike_audit_low_lr/` 和 `formal/spike_audit_clip_half/`。审计只作诊断，不改变 checkpoint 选择或模型门禁；默认模型保持不变。

## MM-E014 / MM-F022 prompt/reward audit

The audit is offline and refuses to overwrite a non-empty output directory:

```bash
.venv/bin/python evaluation/audit_rl_prompt_reward_components.py \
  --source-root results/experiments/rl_update_scale_diagnostic_20260801/formal \
  --output-dir results/experiments/rl_prompt_reward_component_audit_20260801 \
  --top-k 10
```

It requires the three formal run directories and their `microbatch_summaries.jsonl`, `samples.jsonl`, `step_summaries.jsonl`, `validation_history.jsonl`, `selection.json`, and `baseline_validation.json` files. It writes `summary.json`, `run_reports.jsonl`, `prompt_summary.jsonl`, `category_summary.json`, `reward_component_summary.json`, and `report.md`. The output is diagnostic-only and must not be used to bypass KL, quality, or model-promotion gates.

## MM-E015 / MM-F023 corrected telemetry and v2 audit

The trainer correction is covered by the full regression suite. The offline v2 audit is reproducible with a new, empty output directory:

```bash
.venv/bin/python -m py_compile trainer/train_grpo_lite.py evaluation/audit_rl_prompt_reward_components.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python evaluation/audit_rl_prompt_reward_components.py \
  --audit-version v2 \
  --source-root results/experiments/rl_update_scale_diagnostic_20260801/formal \
  --source-manifest results/inputs/rl_data_isolation_train_128_20260801.jsonl \
  --output-dir results/experiments/rl_prompt_reward_component_audit_v2_20260801 \
  --top-k 10
```

The v2 output includes `summary.json`, `run_reports.jsonl`, `cross_condition_prompt_summary.jsonl`, `cross_condition_category_summary.json`, `exposure_denominators.json`, `validator_replay_summary.json`, `reward_component_summary.json`, and `report.md`. It refuses a non-empty output directory and does not rewrite MM-F022. The expected offline acceptance values are source chosen `128/128`, validator replay `1536/1536`, metadata missing `0`, and GPU wall time `0` seconds.

## Corrected GRPO control smoke

The smoke wrapper accepts an explicit independent root so prior MM-E012 artifacts are not reused:

```bash
RL_MICROBATCH_ROOT=results/experiments/rl_corrected_telemetry_smoke_20260801 \
RL_MICROBATCH_TMUX_SESSION=rl_corrected_telemetry_smoke_20260801 \
RL_MICROBATCH_TASK_ID=MM-E015 \
RL_MICROBATCH_HARD_LIMIT_SECONDS=1800 \
bash scripts/run_rl_microbatch_telemetry.sh --smoke
```

The expected smoke contract is GRPO seed 42, 2 steps, 8 train prompts, 2 validation prompts, 2 generations, accumulation 2, and max generation length 16. Check `run.log`, `microbatch_summaries.jsonl`, `samples.jsonl`, `validation_history.jsonl`, and the reloaded checkpoint before considering any larger experiment. The smoke is diagnostic-only and does not alter the default model.

## MM-E016 / MM-F024 corrected balanced diagnostic

The follow-up is a single GRPO control run and refuses reuse of an existing experiment root:

```bash
RL_MICROBATCH_ROOT=results/experiments/rl_corrected_balanced_diagnostic_20260802 \
RL_MICROBATCH_TMUX_SESSION=rl_corrected_balanced_diagnostic_20260802 \
RL_MICROBATCH_TASK_ID=MM-E016 \
RL_MICROBATCH_HARD_LIMIT_SECONDS=7200 \
bash scripts/run_rl_microbatch_telemetry.sh --diagnostic
```

The run uses seed 42, 4 steps, 128 train prompts, 32 validation prompts, 8 generations, accumulation 8, max sequence length 384, max generation length 128, `eval_every=2`, `checkpoint_every=4`, and `interleave_categories=true`. Run the offline audit afterward:

```bash
.venv/bin/python evaluation/audit_rl_spike_sources.py \
  --experiment-root results/experiments/rl_corrected_balanced_diagnostic_20260802 \
  --output-dir results/experiments/rl_corrected_balanced_diagnostic_20260802/audit_corrected_v1 \
  --run-name grpo_control_seed42_diagnostic_balanced \
  --require-microbatch
```

Expected diagnostic status is `BROAD_SPIKE_DIAGNOSTIC`; it is not a model-promotion result.

## MM-E017 / MM-F025 KL guard diagnostic

The guard wrapper defaults to a dry run and refuses to reuse a non-empty run directory or tmux session. The formal command was:

```bash
RL_GUARD_ROOT=results/experiments/rl_kl_guard_diagnostic_20260802 \
RL_GUARD_TMUX_SESSION=rl_kl_guard_formal1_20260802 \
RL_GUARD_RUN_TAG=_formal1 \
RL_GUARD_TASK_ID=MM-E017 \
RL_GUARD_HARD_LIMIT_SECONDS=3570 \
bash scripts/run_rl_kl_guard_diagnostic.sh --diagnostic
```

The wrapper itself runs the GRPO smoke/formal protocol, records resources every 60 seconds, and invokes the offline audit. The explicit audit command is:

```bash
.venv/bin/python evaluation/audit_rl_kl_guard.py \
  --experiment-root results/experiments/rl_kl_guard_diagnostic_20260802 \
  --output-dir results/experiments/rl_kl_guard_diagnostic_20260802/audit_formal_formal1 \
  --run-name grpo_control_seed42_guarded_formal1 \
  --reference-root results/experiments/rl_corrected_balanced_diagnostic_20260802
```

Expected status for the archived run is `GUARD_UNRESOLVED_BASELINE_RETAINED`. It is an update-scale diagnostic only; it does not promote a checkpoint or change the default model. The smoke/formal outputs, step telemetry, rollback evidence, and audit report are all under `results/experiments/rl_kl_guard_diagnostic_20260802/`.

## MM-E018 / MM-F026 KL guard telemetry smoke

The smoke wrapper is intentionally limited to one GRPO run and refuses to reuse a non-empty root or tmux session:

```bash
RL_GUARD_TELEMETRY_ROOT=results/experiments/rl_kl_guard_telemetry_v2_20260802 \
RL_GUARD_TELEMETRY_TMUX_SESSION=rl_kl_guard_telemetry_smoke_20260802 \
RL_GUARD_TELEMETRY_TASK_ID=MM-E018 \
RL_GUARD_TELEMETRY_HARD_LIMIT_SECONDS=1800 \
bash scripts/run_rl_kl_guard_telemetry_smoke.sh --smoke
```

The attempted run exited before attempt telemetry because of a scalar AdamW digest bug. The corrected implementation is in commit `38e9219997b62574f9f09513917c25e1e9ca4e4b`; the original failed run is retained and must not be overwritten or silently rerun. The incomplete audit is:

```bash
.venv/bin/python evaluation/audit_rl_kl_guard_telemetry.py \
  --experiment-root results/experiments/rl_kl_guard_telemetry_v2_20260802 \
  --output-dir results/experiments/rl_kl_guard_telemetry_v2_20260802/audit_smoke_incomplete \
  --run-name grpo_guard_telemetry_smoke_seed42
```

Expected archived status is `TELEMETRY_INCOMPLETE`; no dtype or optimizer conclusion may be drawn from this failed smoke.

## Corrected E018 retry and audit

The first smoke was retained unchanged. After explicit continuation, the corrected implementation was run once in a new root:

```bash
RL_GUARD_TELEMETRY_ROOT=results/experiments/rl_kl_guard_telemetry_v2_retry_20260802 \
RL_GUARD_TELEMETRY_TMUX_SESSION=rl_kl_guard_telemetry_retry_20260802 \
RL_GUARD_TELEMETRY_TASK_ID=MM-E018 \
RL_GUARD_TELEMETRY_HARD_LIMIT_SECONDS=1800 \
bash scripts/run_rl_kl_guard_telemetry_smoke.sh --smoke
```

The wrapper automatically produced `audit_smoke/summary.json`. Its observed status is `BF16_MEASUREMENT_SENSITIVE`: four attempts had bfloat16 gate rejection, float32 diagnostic acceptance, and exact rollback verification. No optimizer step or checkpoint was retained. The retry root is independent of both the first E018 root and `rl_kl_guard_diagnostic_20260802`.

## MM-E019 / MM-F027 true-fp32-copy measurement diagnostic

The optional full-float32 diagnostic is enabled only with `RL_GUARD_TELEMETRY_FULL_FP32=1`; the default wrapper path is unchanged:

```bash
RL_GUARD_TELEMETRY_ROOT=results/experiments/rl_kl_measurement_precision_20260802 \
RL_GUARD_TELEMETRY_TMUX_SESSION=rl_kl_measurement_precision_20260802 \
RL_GUARD_TELEMETRY_TASK_ID=MM-E019 \
RL_GUARD_TELEMETRY_HARD_LIMIT_SECONDS=1800 \
RL_GUARD_TELEMETRY_FULL_FP32=1 \
bash scripts/run_rl_kl_guard_telemetry_smoke.sh --smoke
```

The dedicated audit is:

```bash
.venv/bin/python evaluation/audit_kl_measurement_precision.py \
  --experiment-root results/experiments/rl_kl_measurement_precision_20260802 \
  --output-dir results/experiments/rl_kl_measurement_precision_20260802/precision_audit \
  --run-name grpo_guard_telemetry_smoke_seed42
```

Observed status: `BF16_AUTOCAST_SENSITIVE`. The no-autocast bfloat16-weight measurement and detached true-float32 measurement matched on all four attempts; neither changes the production gate in this diagnostic.

## MM-E020 / MM-F028 independent reference-KL audit

Run the offline-only audit from the repository root:

~~~
bash scripts/run_reference_kl_semantics_audit.sh --dry-run
bash scripts/run_reference_kl_semantics_audit.sh
~~~

The wrapper sets CUDA_VISIBLE_DEVICES="" and refuses to overwrite a non-empty output directory. Results are written to results/experiments/rl_reference_kl_semantics_audit_20260802/. The audit validates formula, completion mask, token-weighted aggregation, persisted gate/rollback consistency, and source-path alignment. It does not claim token-level replay because the E019 artifact does not contain token-level log-probabilities and masks.

## MM-E021 / MM-F029 corrected KL token replay smoke

Use a new root only; the completed smoke root must not be reused:

~~~bash
bash scripts/run_corrected_kl_replay_smoke.sh --dry-run
bash scripts/run_corrected_kl_replay_smoke.sh --smoke
~~~

The wrapper runs one GRPO seed-42 smoke under tmux, records resources every 60 seconds, refuses non-empty roots/sessions, and invokes the offline audit after training. The completed artifact is `results/experiments/rl_corrected_kl_replay_smoke_20260802/`; the final audit is:

~~~bash
PYTHONPATH=. .venv/bin/python evaluation/audit_kl_token_replay.py \
  --experiment-root results/experiments/rl_corrected_kl_replay_smoke_20260802 \
  --output-dir results/experiments/rl_corrected_kl_replay_smoke_20260802/replay_audit_v2 \
  --run-name grpo_corrected_kl_replay_smoke_seed42
~~~

Expected final status is `TOKEN_REPLAY_VALIDATED`. The audit is offline and diagnostic-only; it does not select a checkpoint or alter the production bfloat16 gate. The smoke used 30 GPU wall seconds, accepted no optimizer step, and retained the baseline.

## MM-E022 / MM-F030 corrected post-step gate smoke

The corrected gate is opt-in; omitting `--post-step-kl-gate-mode` preserves the legacy bfloat16 gate. The completed root must not be reused:

~~~bash
bash scripts/run_corrected_kl_gate_smoke.sh --dry-run
bash scripts/run_corrected_kl_gate_smoke.sh --smoke
~~~

The wrapper launches one GRPO seed-42 smoke under tmux, enforces the 1800-second and 80-GiB limits, records resources, and invokes the offline audit. To reproduce only the audit into a new empty directory:

~~~bash
PYTHONPATH=. .venv/bin/python evaluation/audit_corrected_kl_gate.py \
  --experiment-root results/experiments/rl_corrected_kl_gate_smoke_20260802 \
  --output-dir results/experiments/rl_corrected_kl_gate_smoke_20260802/corrected_gate_audit_recheck \
  --run-name grpo_corrected_kl_gate_smoke_seed42
~~~

The archived result is `CORRECTED_GATE_ACCEPTED_2_STEPS_DIAGNOSTIC`: both first attempts passed the active FP32/no-autocast mean-KL gate, state continuity and token replay passed, and the step-2 checkpoint reloaded. Expected warnings include `CHECKPOINT_SELECTION_SMOKE_ONLY`, `LEGACY_BF16_SHADOW_DISAGREEMENT`, and `PRESTEP_PRECISION_DIVERGENCE`. The 2-prompt validation is plumbing-only, and the checkpoint must not be promoted.

## MM-E023 / MM-F031 same-token pre-step precision diagnostic

The completed root must not be reused. The production loss remains unchanged; FP32 loss and gradients are shadow telemetry only:

~~~bash
bash scripts/run_prestep_precision_smoke.sh --dry-run
bash scripts/run_prestep_precision_smoke.sh --smoke
~~~

To rerun only the offline audit, choose a new empty output directory:

~~~bash
PYTHONPATH=. .venv/bin/python evaluation/audit_prestep_precision.py \
  --experiment-root results/experiments/rl_prestep_precision_smoke_20260802 \
  --output-dir results/experiments/rl_prestep_precision_smoke_20260802/prestep_precision_audit_recheck \
  --run-name grpo_prestep_precision_smoke_seed42
~~~

Expected status is `TRAINING_AUTOCAST_PRECISION_SENSITIVE`. Four pre-step replay rows must validate against the persisted token terms, four micro-batch links, production-gradient isolation, two accepted gate attempts, twelve post-step replay rows, state continuity, and the reloaded step-2 checkpoint. The expected limitation warning is `ZERO_ADVANTAGE_KL_ONLY_SMOKE`; this run cannot establish nonzero-advantage clipping behavior or model quality.
## MM-E024 / MM-F032 opt-in FP32 training-forward smoke

Run from `.`:

```bash
bash scripts/run_fp32_training_forward_smoke.sh --dry-run
bash scripts/run_fp32_training_forward_smoke.sh --smoke
```

The wrapper creates the isolated root `results/experiments/rl_fp32_training_forward_smoke_20260802/`, uses tmux, records command/commit/environment/GPU/disk/wall time/exit code, and refuses to reuse a non-empty root or session. The active training forward and post-step gate are both `fp32_no_autocast`; the legacy bfloat16 path remains a shadow.

The final offline audit is:

```bash
PYTHONPATH=. .venv/bin/python evaluation/audit_fp32_training_forward.py \
  --experiment-root results/experiments/rl_fp32_training_forward_smoke_20260802 \
  --output-dir results/experiments/rl_fp32_training_forward_smoke_20260802/fp32_training_forward_audit_v2 \
  --run-name grpo_fp32_training_forward_smoke_seed42 \
  --expected-training-forward-mode fp32_no_autocast
```

Expected status is `FP32_TRAINING_FORWARD_ACCEPTED_2_STEPS_DIAGNOSTIC`. The smoke has a required limitation warning: `ZERO_ADVANTAGE_NO_PARAMETER_UPDATE`; the nonzero-advantage contract is covered by `tests/test_fp32_training_forward.py`, not by this model run.

## MM-E025 / MM-F033 deterministic nonzero-advantage contract

This stage is offline-only and refuses to reuse a non-empty experiment root:

```bash
bash scripts/run_nonzero_advantage_contract.sh --dry-run
bash scripts/run_nonzero_advantage_contract.sh --run
```

The fixture is `results/inputs/rl_nonzero_advantage_contract_fixture_20260802.json`; the isolated output is `results/experiments/rl_nonzero_advantage_contract_20260802/`. The wrapper disables CUDA, records command/commit/environment/disk/exit code/wall time, and invokes `evaluation/audit_nonzero_advantage_contract.py`. Expected status is `NONZERO_ADVANTAGE_CONTRACT_PASS`: GRPO and CISPO must both show nonzero advantages, the expected clipped-token counts, matching production/diagnostic loss and KL, and finite nonzero FP32 and bfloat16-quantized gradients. This contract is diagnostic-only and does not select checkpoints or alter the default model.

## MM-E026 / MM-F034 controlled nonzero-advantage smoke

The active-path smoke is isolated and default-off:

```bash
bash scripts/run_controlled_nonzero_advantage_smoke.sh --dry-run
bash scripts/run_controlled_nonzero_advantage_smoke.sh --smoke
```

It creates `results/experiments/rl_controlled_nonzero_advantage_smoke_20260802/`, refuses a non-empty root or tmux session, uses one GRPO seed 42 task, and applies the diagnostic-only controlled pattern `[1.0, 0.0]`. The active training-forward and post-step gate are `fp32_no_autocast`; the legacy bfloat16 path remains shadow telemetry. The smoke records replay, samples, micro-batches, guard attempts, state digests, resources, exit code and wall time.

The corrected offline audit is:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python evaluation/audit_controlled_nonzero_advantage.py \
  --experiment-root results/experiments/rl_controlled_nonzero_advantage_smoke_20260802 \
  --output-dir results/experiments/rl_controlled_nonzero_advantage_smoke_20260802/active_path_audit_v2 \
  --run-name grpo_controlled_nonzero_advantage_smoke_seed42 \
  --expected-pattern 1.0,0.0 \
  --expected-training-forward-mode fp32_no_autocast \
  --expected-steps 2
```

Expected status is `CONTROLLED_NONZERO_ADVANTAGE_ACTIVE_PATH_PASS_2_STEPS`. This status validates only the active update/telemetry/checkpoint path. The injected reward is not a quality signal; do not promote its checkpoint or infer validator improvement.

## MM-E027 / MM-F035 natural rule-reward corrected smoke

Run the isolated natural-reward smoke with the controlled override absent:

```bash
bash scripts/run_natural_rule_reward_smoke.sh --dry-run
bash scripts/run_natural_rule_reward_smoke.sh --smoke
```

The wrapper creates `results/experiments/rl_natural_rule_reward_smoke_20260802/`, records the natural `rule_reward` source, uses one GRPO seed 42 task with the opt-in FP32/no-autocast training-forward and gate, and refuses to reuse a non-empty root or tmux session. The run keeps full replay, sample, micro-batch, guard, state, checkpoint, validation and resource artifacts.

The corrected offline audit is:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python evaluation/audit_natural_rule_reward_smoke.py \
  --experiment-root results/experiments/rl_natural_rule_reward_smoke_20260802 \
  --output-dir results/experiments/rl_natural_rule_reward_smoke_20260802/natural_rule_reward_audit_v2 \
  --run-name grpo_natural_rule_reward_smoke_seed42 \
  --expected-training-forward-mode fp32_no_autocast \
  --expected-steps 2
```

Expected status for the observed run is `NATURAL_RULE_REWARD_ZERO_ADVANTAGE_DIAGNOSTIC`. This means natural reward metadata and the active update path are valid, but the short run produced no nonzero group advantage; it is not a quality or promotion result.

### MM-E028 / MM-F036 natural reward diversity audit

Run the offline wrapper from the repository root:

```bash
bash scripts/run_natural_reward_diversity_audit.sh --dry-run
bash scripts/run_natural_reward_diversity_audit.sh --run
```

The wrapper disables CUDA and refuses to reuse a non-empty output root. The audit reads the E027 natural smoke and reports E010/E009 legacy artifacts separately. Results are written to `results/experiments/rl_natural_reward_diversity_audit_20260802/audit/summary.json`, `report.md`, and `input_manifest.json`. The observed status is `NATURAL_REWARD_DIVERSITY_AUDIT_COLLAPSE_CONFIRMED`; this is diagnostic evidence only and does not authorize GPU training or model promotion.

### MM-E029 / MM-F037 natural reward input/output coverage audit

Run:

```bash
bash scripts/run_natural_reward_coverage_audit.sh --dry-run
bash scripts/run_natural_reward_coverage_audit.sh --run
```

The wrapper reads `results/inputs/rl_data_isolation_train_128_20260801.jsonl` and the E027 natural smoke, disables CUDA, refuses to reuse a non-empty output root, and records resource state. Results are written to `results/experiments/rl_natural_reward_coverage_audit_20260802/audit/summary.json`, `report.md`, and `input_manifest.json`. The observed status is `CURRENT_GENERATION_SIGNAL_COVERAGE_INSUFFICIENT_DIAGNOSTIC`; deterministic probes are explicitly not model-generation evidence.
### MM-E030 / MM-F038 balanced natural-reward coverage smoke

The selector and wrapper are:

```bash
PYTHONPATH=. .venv/bin/python evaluation/prepare_balanced_coverage_manifests.py \
  --train-source results/inputs/rl_data_isolation_train_128_20260801.jsonl \
  --validation-source results/inputs/rl_data_isolation_validation_32_20260801.jsonl \
  --output-root results/experiments/rl_balanced_reward_coverage_smoke_20260802/inputs \
  --seed 42
bash scripts/run_balanced_reward_coverage_smoke.sh --dry-run
bash scripts/run_balanced_reward_coverage_smoke.sh --smoke
```

The wrapper refuses an existing non-empty experiment root, uses one GRPO seed-42 task under tmux, polls resources at 60-second intervals, and automatically runs `evaluation/audit_natural_reward_coverage.py`. The observed artifact has 8/8 category coverage and 32 samples; the audit status is `CURRENT_GENERATION_SIGNAL_VARIABILITY_OBSERVED_DIAGNOSTIC`, with validator pass `2/32` and validation baseline `0/8`. This is diagnostic only; do not use the selected checkpoint as a promoted model.
### MM-E031 / MM-F039 balanced output-quality audit

Run the CUDA-disabled audit from the repository root:

```bash
bash scripts/run_balanced_output_quality_audit.sh --dry-run
bash scripts/run_balanced_output_quality_audit.sh --run
```

The wrapper refuses a non-empty output root and replays `results/experiments/rl_balanced_reward_coverage_smoke_20260802/` into `results/experiments/rl_balanced_output_quality_audit_20260802/audit/`. It writes `summary.json`, `category_summary.json`, `sample_diagnostics.jsonl`, `input_manifest.json` and `report.md`. The observed status is `OUTPUT_QUALITY_SIGNAL_SPARSE_DIAGNOSTIC`: coverage is complete and replay is exact, but validator pass is `2/32` and max-length hits are `20/32`. This is diagnostic only and does not authorize GPU training or model promotion.
### MM-E032 / MM-F040 release preflight

Run from the repository root after confirming that the v2 isolated manifests exist:

```bash
RL_RELEASE_GATE_ROOT=results/experiments/rl_release_gate_20260802_retry1 \
RL_RELEASE_GATE_TMUX_SESSION=rl_release_gate_preflight_retry1_20260802 \
bash scripts/run_rl_release_gate.sh --dry-run
RL_RELEASE_GATE_ROOT=results/experiments/rl_release_gate_20260802_retry1 \
RL_RELEASE_GATE_TMUX_SESSION=rl_release_gate_preflight_retry1_20260802 \
bash scripts/run_rl_release_gate.sh --preflight
```

The preflight command uses GRPO seed 42, four steps, eight generations, 128 train prompts and 32 validation prompts. It requires `training_forward_mode=fp32_no_autocast` and `post_step_kl_gate_mode=fp32_no_autocast`; the active gate is still targeted at `0.005`, while bfloat16 remains shadow telemetry. The audited result was `PREFLIGHT_PASS`.

### MM-E033 / MM-F041 formal matrix and frozen evidence

After and only after the preflight decision is `PREFLIGHT_PASS`, run the formal matrix in its own non-empty root:

```bash
RL_RELEASE_GATE_ROOT=results/experiments/rl_release_gate_20260802_retry1 \
RL_RELEASE_GATE_TMUX_SESSION=rl_release_gate_formal_retry1_20260802 \
bash scripts/run_rl_release_gate.sh --formal
```

Then run the offline formal audit and the retained frozen evidence:

```bash
.venv/bin/python evaluation/audit_rl_release_gate.py \
  --phase formal \
  --run-dir results/experiments/rl_release_gate_20260802_retry1/formal \
  --output-dir results/experiments/rl_release_gate_20260802_retry1/formal/audit_v2
RL_RELEASE_GATE_ROOT=results/experiments/rl_release_gate_20260802_retry1 \
RL_RELEASE_GATE_TMUX_SESSION=rl_release_gate_frozen_retry1_20260802 \
bash scripts/run_rl_release_gate.sh --frozen-eval
```

The wrapper refuses a non-empty root/run directory or reused tmux session, records command/commit/environment/resource/exit metadata, runs one GPU task at a time, and preserves every checkpoint, validation history and failure log. Frozen evaluation is evidence only and does not participate in checkpoint selection or promotion. C-Eval is not part of this reproduction path.
### MM-E034 / MM-F042 quality signal repair

Run the audit in a fresh root:

`QUALITY_SIGNAL_REPAIR_ROOT=results/experiments/quality_signal_repair_20260803_retry2 bash scripts/run_quality_signal_repair.sh --audit`

The audit is expected to pass the native-v2 input contract and the prepare phase is expected to exit `4` when the fixed selector finds fewer than 96 rows for a target category. In the recorded run, `conciseness` had `80/96`; `quality_repair_train.jsonl`, baseline evaluation, SFT and candidate evaluation were intentionally not produced. Do not rerun with a mixed source or reduced quota without a new plan.
### MM-E034 / MM-F042 authorized conciseness supplement and SFT smoke

The original train manifest is preserved. Prepare the deterministic supplement and augmented manifest:

` .venv/bin/python dataset/alignment_v2/prepare_conciseness_supplement_20260803.py --supplement-output results/inputs/quality_signal_repair_conciseness_supplement_16_20260803.jsonl --augmented-output results/inputs/quality_signal_repair_native_v2_train_manifest_1016_20260803.jsonl --summary-output results/inputs/quality_signal_repair_conciseness_supplement_16_20260803.json`

Run the isolated audit and smoke with `QUALITY_SIGNAL_REPAIR_EXPECTED_NATIVE_TRAIN_COUNT=1016` and `QUALITY_SIGNAL_REPAIR_TRAIN_MANIFEST=results/inputs/quality_signal_repair_native_v2_train_manifest_1016_20260803.jsonl`. The recorded result is `QUALITY_REPAIR_PASS_DIAGNOSTIC`; it does not authorize automatic RL or default-model replacement.
### MM-E035 / MM-F043 corrected-GRPO smoke

Use a new root and the isolated candidate checkpoint:

`QUALITY_REPAIR_GRPO_ROOT=results/experiments/quality_repair_corrected_grpo_smoke_20260803 QUALITY_REPAIR_GRPO_TMUX_SESSION=quality_repair_corrected_grpo_20260803 QUALITY_REPAIR_GRPO_FROM_WEIGHT=quality_repair_sft_seed42 QUALITY_REPAIR_GRPO_MODEL_DIR=results/experiments/quality_signal_repair_20260803_augmented_retry1/sft_repair_seed42/out scripts/run_quality_repair_corrected_grpo_smoke.sh --smoke`

The wrapper uses a two-step GRPO smoke, FP32/no-autocast training and active post-step gate, while retaining bfloat16 shadow, token replay, state digests and checkpoint reload. The recorded audit is `CORRECTED_GATE_ACCEPTED_2_STEPS_DIAGNOSTIC`; it is plumbing-only, with a two-prompt validation result of `0/2`. Do not use the selected smoke checkpoint as a default model or as evidence for formal RL adoption.
### MM-F044 precision attribution audit

Run the offline audit into a fresh subdirectory:

`CUDA_VISIBLE_DEVICES='' .venv/bin/python evaluation/audit_prestep_precision.py --experiment-root results/experiments/quality_repair_corrected_grpo_smoke_20260803 --output-dir results/experiments/quality_repair_corrected_grpo_smoke_20260803/prestep_precision_audit --run-name grpo_quality_repair_corrected_seed42`

The recorded result is `TRAINING_AUTOCAST_PRECISION_SENSITIVE`; the fixed post-step comparison further classifies it as `BF16_MEASUREMENT_SENSITIVE`. This is an offline diagnostic only. Resolve or explicitly bound the precision semantics before any 4-step or formal RL expansion.

### MM-E036 / MM-F045 explicit precision contract smoke

Run the dry-run first, then the single smoke from the repository root:

```bash
scripts/run_precision_contract_smoke.sh --dry-run
scripts/run_precision_contract_smoke.sh --smoke
```

The wrapper uses a fresh root `results/experiments/rl_precision_contract_smoke_20260803/`, the isolated Alignment v2 train/validation manifests, and the quality-repair candidate checkpoint. It requires `precision_contract_mode=no_autocast_v1`, `training_forward_mode=fp32_no_autocast`, `post_step_kl_gate_mode=fp32_no_autocast`, pre-step loss replay, token replay, full-FP32 shadow and micro-batch telemetry. It refuses an existing root or tmux session, records resources every 60 seconds, enforces an 1800-second wall limit, and runs `evaluation/audit_precision_contract.py` after the smoke.

The recorded audit is `PRECISION_CONTRACT_PASS_WITH_BF16_SHADOW_WARNING`: 2/2 steps accepted, active/full-FP32 KL matched, replay and checkpoint reload passed, and the bfloat16 shadow remained warning-only. This is a diagnostic contract result; do not treat it as quality evidence or model promotion.
### MM-E037 / MM-F046 four-step precision-contract diagnostic

Run the dry-run before the single diagnostic:

```bash
scripts/run_precision_contract_4step.sh --dry-run
scripts/run_precision_contract_4step.sh --diagnostic
```

The wrapper creates the fresh root `results/experiments/rl_precision_contract_4step_20260803/`, refuses an existing root or tmux session, uses the isolated Alignment v2 manifests and quality-repair candidate, records resources every 60 seconds, and enforces an 1800-second wall limit. It runs GRPO seed 42 for four steps with the same `no_autocast_v1` active loss/gate contract as the two-step smoke. The recorded audit is `PRECISION_CONTRACT_PASS_WITH_BF16_SHADOW_WARNING_4_STEPS`; use the final audit directory `precision_contract_audit_final/` for the preserved run because the first wrapper version pre-created its audit directory and exited before auditing.
### MM-E038 / MM-F047 offline precision/quality audit

Run the audit with CUDA disabled and a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python evaluation/audit_precision_quality_4step.py \
  --experiment-root results/experiments/rl_precision_contract_4step_20260803 \
  --output-dir results/experiments/rl_precision_contract_4step_20260803/precision_quality_audit \
  --run-name grpo_precision_contract_4step_seed42 \
  --expected-steps 4
```

The audit does not load model weights. It checks active/full-FP32 KL agreement, pre-step precision telemetry, state/digest continuity, checkpoint reload, sample linkage, reward replay and the limited quality scope. The recorded status is `PRECISION_DIVERGENCE_PERSISTS_QUALITY_SCOPE_LIMITED_DIAGNOSTIC`; it is not a promotion or RL-quality result.
### MM-E039 / MM-F048 quality evidence boundary audit

Run with CUDA disabled and a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python evaluation/audit_quality_evidence_boundary.py \
  --quality-summary results/experiments/quality_signal_repair_20260803_augmented_retry1/quality_decision/summary.json \
  --input-summary results/experiments/quality_signal_repair_20260803_augmented_retry1/input_audit/summary.json \
  --rl-summary results/experiments/rl_precision_contract_4step_20260803/precision_quality_audit/summary.json \
  --output-dir results/experiments/rl_precision_contract_4step_20260803/quality_evidence_boundary_audit
```

The audit is artifact-only and rejects non-empty output directories. It records the SFT quality boundary separately from the RL telemetry boundary. Expected status is `QUALITY_EVIDENCE_BOUNDARY_DEFINED_DIAGNOSTIC`; formal RL readiness and automatic GPU start remain false.
### MM-E040 / MM-F049 corrected-GRPO quality evidence diagnostic

Run the fresh-root dry-run and diagnostic wrapper:

```bash
scripts/run_rl_quality_evidence_diagnostic.sh --dry-run
scripts/run_rl_quality_evidence_diagnostic.sh --diagnostic
```

The wrapper uses the isolated 128/32 Alignment v2 manifests and the isolated quality-repair candidate. It runs GRPO seed 42 for four steps with eight generations, eight-step accumulation, full balanced 32-row validation, the opt-in `no_autocast_v1` contract and active FP32/no-autocast gate. It refuses an existing root or run directory, records tmux/resource/command metadata, enforces a 3600-second limit, and runs both precision-contract and quality-evidence audits.

Expected quality audit: `QUALITY_EVIDENCE_DIAGNOSTIC_COMPLETE`. In this run the source and selected checkpoint both scored `19/32`; this is directional evidence only and does not authorize formal RL or model replacement.
### MM-E041 / MM-F050 zero-gain failure attribution audit

Run the offline audit from a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python evaluation/audit_rl_quality_failure_attribution.py \
  --experiment-root results/experiments/rl_quality_evidence_corrected_grpo_20260803 \
  --run-name grpo_quality_evidence_seed42 \
  --output-dir results/experiments/rl_quality_evidence_corrected_grpo_20260803/quality_failure_attribution_audit_v3 \
  --selected-step 2
```

The audit is artifact-only and rejects non-empty output directories. Expected status is `QUALITY_FAILURE_ATTRIBUTION_COMPLETE`; it compares source SFT and selected step-2 validation item IDs, aggregates failure reasons and reward-component coverage, and never starts a GPU task. The expected directional result is `19/32` versus `19/32`, with `13` stable failures and `19` stable passes.
### MM-E042 / MM-F051 reward-input and validator-contract audit

Run from a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python evaluation/audit_rl_reward_input_contract.py \
  --experiment-root results/experiments/rl_quality_evidence_corrected_grpo_20260803 \
  --run-name grpo_quality_evidence_seed42 \
  --output-dir results/experiments/rl_reward_input_contract_audit_20260803
```

The audit refuses a non-empty output directory, validates the resolved manifest, replays chosen and generated responses through both `validate_record` and `rule_reward`, checks category-component routing, and reports prompt/family coverage and termination semantics. Expected status is `REWARD_INPUT_COVERAGE_LIMITED_DIAGNOSTIC`; it must not start a GPU task.
### MM-E043 / MM-F052 Output-to-Validator mapping audit

Run from a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python evaluation/audit_rl_output_validator_mapping.py \
  --experiment-root results/experiments/rl_quality_evidence_corrected_grpo_20260803 \
  --run-name grpo_quality_evidence_seed42 \
  --output-dir results/experiments/rl_output_validator_mapping_audit_20260803
```

The audit refuses non-empty output directories, hashes the manifest/samples/validator/rule files, replays chosen and generated responses, and writes `summary.json`, `failure_cases.jsonl`, `input_manifest.json` and `report.md`. Expected status is `OUTPUT_VALIDATOR_MAPPING_CONSISTENT_LIMITED_DIAGNOSTIC`; no model is loaded and no GPU task starts.

## MM-E044 / MM-F053 category-weighting audit

From `.`, use the project environment and keep CUDA disabled:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python evaluation/audit_rl_category_weighting.py \
  --experiment-root results/experiments/rl_quality_evidence_corrected_grpo_20260803 \
  --run-name grpo_quality_evidence_seed42 \
  --output-dir results/experiments/rl_category_weighting_audit_20260803_v2
```

The audit refuses a non-empty output directory. Inputs are the resolved 128-row native-v2 train manifest, `samples.jsonl`, and `step_summaries.jsonl`. It writes `summary.json`, `group_summaries.jsonl`, and `report.md`; it does not load weights or start a GPU task. The archived implementation, test, summary and group-summary hashes are recorded in `results/task_state.json`.

## MM-E045 / MM-F054 error-driven SFT and preference smoke

From ., run the data audit and then a fresh-root smoke with scripts/run_error_driven_preference_repair.sh. The wrapper evaluates baseline, error-driven SFT, DPO and SimPO on the same 160 validation and 32 release rows. DPO save_interval equals max_steps so short runs produce a strict-reloadable checkpoint. Retry5 is QUALITY_METHODS_NOT_MET_NO_MODEL_CHANGE and does not authorize corrected-GRPO.


## MM-E047 / MM-F056：复现入口

1. 进入 `.` 并确认 `results/experiments/error_driven_preference_repair_20260803_retry6/method_comparison/summary.json` 的质量门禁结果。
2. 使用独立的 error-driven SFT 候选和 `results/inputs/rl_data_isolation_train_128_20260801.jsonl`、`results/inputs/rl_data_isolation_validation_32_20260801.jsonl`。
3. 运行 `scripts/run_rl_quality_evidence_diagnostic.sh --diagnostic` 可复现 4-step corrected-GRPO 诊断；审计输出位于该实验根目录的 `precision_contract_audit/` 与 `quality_evidence_audit/`。

该命令只用于诊断，不代表正式 RL 放行；正式六 seed、多算法和默认权重替换仍需新的明确批准。
