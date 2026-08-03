# 成本与资源报告

## 现场资源

- Google Cloud Compute Engine，NVIDIA L4 24GB。
- F 阶段继续执行前磁盘约 113GB 可用，未触发 30GB 停止线。
- 每个长 GPU 任务均通过 `scripts/run_experiment.sh` + tmux 保存 start/end、GPU、磁盘、环境、stdout/stderr 和失败状态。

## 成本口径

本机仓库没有 Cloud Billing export、账单 API 返回或可验证的 L4 USD 单价。因此本报告不伪造美元金额；`estimated_cost` 对无账单证据的实验保持 `null`，并用 wrapper wall time、GPU 型号和 checkpoint 大小作为工程成本代理。用户已明确允许使用 Google Cloud 礼金进行 Gemini 调用，这说明本轮可继续执行，但不等于本地已测得美元成本。

精确 wrapper wall time 汇总：

`results/experiments/unified_sft_v2_20260731/unified_metrics.json`

图表：

`results/experiments/unified_sft_v2_20260731/plots/experiment_wall_time.png`

## 成本控制

- 所有 GPU 训练单卡串行；训练前检查 `nvidia-smi`、`df -h`、进程和 tmux。
- C-E 使用 smoke → pilot；DPO、SimPO、Reward、GRPO、CISPO 均使用独立输出目录，不覆盖旧权重。
- 本轮新增 RL 扩展为 GRPO/CISPO 各 8 steps、每组 8 generations，并各自完成 100 条冻结集生成与 validator 评分；产物目录和 wrapper wall time 均独立记录，未覆盖 tiny pilot。
- Gemini 调用有独立 experiment id；原始评审响应不加入训练集。
- 本轮 Gemini 扩展实际完成 4×100 条冻结 A/B 评审：DPO、SimPO pilot、SimPO full seed=42，以及 SimPO full seed=43 独立复核；另完成 C003 on-policy 训练集 128 条和验证集 32 条排名（含 4 条 smoke）。wrapper wall time 分别记录在对应 experiment 目录。美元账单仍没有本地可验证 export，因此不把礼金余额换算成伪造 USD。
- F 的 `reproduce_all.sh` 默认 `--dry-run`，不会无提示启动昂贵训练。

## 2026-08-01 新增资源记录

- MM-E009 使用 `scripts/run_rl_suite.sh --full`，GRPO/CISPO 六个 seed 串行运行；每个 run.log 记录 exit code、GPU、磁盘、git commit 和环境 hash。启动基线约 108 GiB 可用磁盘，未触发资源门禁。
- MM-F015 使用 `scripts/run_ceval_subset.sh`，8 个模型串行评测，最终 run exit code 为 0；固定 revision 和 manifest hash 记录在 `results/experiments/ceval_subset_20260801_v3/source.json`。
- 账单 USD 仍没有可验证的 Cloud Billing export，因此不根据礼金余额推算或伪造金额；只报告可审计的 wall-time/log/resource 代理。

## 2026-08-01 MM-E010 / MM-F017 资源记录

- checkpoint 回载修复后的六个正式 RL run GPU wall time 分别为 200、197、200、203、202、175 秒，合计 1177 秒；七个冻结集 generation 分别为 97、97、96、95、98、96、95 秒，合计 674 秒。可审计合计为 1851 秒，约 30 分 51 秒，低于 28800 秒（8 小时）硬上限。
- 六个训练 run 和七个冻结 generation 均 exit code 0；每个独立目录保留 command、commit `14076033f25fc7dfa35403f2d7beccb46ae43d5c`、environment hash `e55aadc2a3df4ab553c1d3fae8df57f1fa79f4108432e34a02296dfd11cace79`、GPU、磁盘、resource monitor 和 wall time。正式训练结束后 GPU 为 0 MiB，服务器保持 `RUNNING`。
- 修正版根目录内 70 个 JSON/JSONL 文件全部可解析且无非有限数；六个训练 run 均有 step summary、validation history、selection、checkpoint 和 audit 记录。旧 `rl_data_isolation_20260801`、validation coverage 和 smoke 目录均未覆盖。
- checkpoint 回载修复 smoke 与独立 `max_steps=0` 复核共 2 个 wrapper，GPU wall time 27 秒，均 exit code 0；只写入独立 smoke 目录，不覆盖旧权重。
- 服务器在实验完成后保持 `RUNNING`，没有执行关机或释放实例操作。真实 USD 成本仍不可由本地日志推导，礼金余额不换算为伪造账单。

## 2026-08-01 MM-E011 / MM-F018 资源记录

- 稳定性诊断使用新根目录 `results/experiments/rl_stability_diagnostic_20260801/`，GRPO/CISPO 各 3 个条件，单卡串行；六个正式 run 均 exit code 0，GPU wall time 分别为 200、202、163、201、199、160 秒，合计 1125 秒；GRPO smoke 为 13 秒。
- 每个 run.log 保留完整 command、commit `b37632ed8519ed92c5b1a3e69b6992fa14d638b3`、environment hash `959a641cdc0e988066d3646bb978fb94fdd4ff5ab967736f04d6e1a4f3c09569`、NVIDIA L4、磁盘/GPU before-after、exit code、finish time 和 wall time。统一 wrapper 设置 7200 秒硬上限，未触发预算停止。
- 六个正式目录均有 step telemetry、validation history、step checkpoint、selected output 和 selection；稳定性审计汇总 `all_json_finite=true`，没有覆盖 E009/E010 或既有权重/结果目录。
- 正式 run 结束 GPU 为 0 MiB，磁盘约 95 GiB 可用；服务器保持 `RUNNING`，没有执行关机或释放实例操作。
- 本轮没有 C-Eval、100 条冻结集或 Gemini 调用；美元账单仍没有可验证 Cloud Billing export，因此不根据礼金余额换算或伪造 USD 成本。

## 2026-08-01 MM-F019 资源记录

- MM-F019 只读取既有六个 run 的 JSON/JSONL 并生成离线审计，未启动 GPU；新增 GPU wall time 为 0 秒。
- 输出写入 `results/experiments/rl_stability_diagnostic_20260801/spike_source_audit_v2/`，不覆盖原稳定性审计目录。服务器保持 `RUNNING`，GPU 仍为空闲。

## 2026-08-01 MM-E012 / MM-F020 资源记录

- micro-batch smoke 运行 13 秒；首个覆盖不完整的 4-step formal run 运行 64 秒并保留；修正类别交错后的 balanced formal run 运行 71 秒。三次 GPU wall time 合计 148 秒，远低于本轮 4 小时硬上限。
- balanced run 和两次训练 run.log 均保留 command、commit `6e450fab87a296d44192fc8fc8d07285047d2a95`、environment hash、NVIDIA L4、磁盘/GPU before-after、exit code、finish time 和 wall time。离线 spike-source audit 未启动 GPU，新增 GPU wall time 为 0 秒。
- 全部新结果位于 `results/experiments/rl_microbatch_telemetry_20260801/`，覆盖不完整的 run 作为审计证据保留；`audit_balanced_v2/` 是最终权威离线审计输出，不覆盖 smoke、formal 或旧实验目录。
- 训练完成后 GPU 为 0 MiB，磁盘约 94 GiB 可用，tmux 任务已结束，服务器保持 `RUNNING`。未执行关机或释放实例操作。
- 由于没有可验证 Cloud Billing export，仍不把礼金余额换算成美元；本轮成本只报告 GPU wall time 和资源状态。

## 2026-08-01 MM-E013 / MM-F021 资源记录

- 更新尺度 smoke 为 13 秒；control、low_lr、clip_half formal 分别为 119、119、118 秒，GPU wall time 合计 369 秒，低于本轮 7200 秒预算。
- 三个 run.log 均记录 command、commit `3be1c740327db117cc114b387d1f750ca33cc114`、environment hash `8fd2a63c227d9d0ca50522562b1492a765557c6ab1ad0d083aeb719ff7d2efd5`、NVIDIA L4、磁盘/GPU before-after、exit code、finish time 和 wall time。
- stability 与 spike audit 均为离线步骤，新增 GPU wall time 为 0 秒；审计器修正 commit 为 `a874cfb91ff98be5f7f9f60d51d03442daa791a4`。
- 全部输出位于 `results/experiments/rl_update_scale_diagnostic_20260801/`，没有覆盖 `rl_microbatch_telemetry_20260801`、`rl_stability_diagnostic_20260801` 或既有权重。
- formal 结束后 GPU 为 0 MiB，磁盘约 92 GiB 可用，服务器保持 `RUNNING`；没有执行关机或释放实例操作。

## 2026-08-01 MM-E014 / MM-F022 offline audit

- GPU wall time: `0` seconds; no new GPU task was started.
- Source runs: the previously completed three update-scale formal runs, 64 micro-batches and 512 samples per run.
- Audit output: `results/experiments/rl_prompt_reward_component_audit_20260801/`.
- `all_json_finite=true`; sample linkage complete for all 1,536 samples.
- The audit used the existing L4 host only for read-only/offline processing. The server remained `RUNNING`, GPU returned/was observed at `0 MiB`, and approximately `92 GiB` disk was available after the audit.

No USD estimate is emitted because no new GPU or billing export was involved. Existing wrapper wall times remain the auditable cost proxy for the source runs.

## 2026-08-01 MM-E015 / MM-F023

- The telemetry correction, regression tests, and v2 validator/input audit were offline; incremental GPU wall time was `0` seconds.
- The audit output is `results/experiments/rl_prompt_reward_component_audit_v2_20260801/`, separate from the existing MM-F022 root. No weights, C-Eval results, or old audit artifacts were overwritten.
- The server remained `RUNNING`; the L4 was observed idle at `0 MiB` after the audit, with approximately `92 GiB` disk available. No shutdown or resource release was performed.
- No USD amount is reported because no new GPU task or verifiable Cloud Billing export was used.

## Corrected GRPO control smoke follow-up

- One isolated smoke used the L4 for `13` GPU wall seconds; the hard safety budget was `1800` seconds and was not approached.
- The wrapper recorded command, commit `32b8ce143f4c71549fc04d669aad84c2c4db64ef`, environment hash, GPU/disk before-after, exit code, and wall time in the independent smoke run log.
- GPU returned to `0 MiB`, disk remained approximately `92 GiB` available, and the server stayed `RUNNING`.

## 2026-08-02 MM-E016 / MM-F024 corrected balanced diagnostic

- One GRPO control diagnostic used the L4 for `80` GPU wall seconds against a hard limit of `7200` seconds.
- The wrapper preserved command, commit `67e80e4a00a8bbb9eca7916750af3211f5defe37`, environment hash `5be16f243a078c7ca9c88b37c7cbb78a71d05c810a38d00f05151297a3d2a1bd`, GPU/disk snapshots, exit code `0`, and finish time in the independent run log.
- After completion the GPU was at `0 MiB`, disk availability was approximately `92 GiB`, and the server remained `RUNNING`. The offline audit added `0` GPU seconds.
- No C-Eval, frozen-set evaluation, Gemini call, or USD billing export was used; no dollar amount is inferred from the credit balance.

## 2026-08-02 MM-E020 / MM-F028 offline audit

- The independent reference-KL semantics audit used 0 GPU wall seconds and did not load a model or expose CUDA.
- The audit consumed no additional cloud training time; it only parsed existing E019 artifacts and source files.
- Server state after the audit: RUNNING, L4 at 0 MiB, approximately 92 GiB available.
- No billing export was available, so no USD amount is inferred.

## 2026-08-02 MM-E019 / MM-F027 measurement precision diagnostic

- The isolated true-fp32-copy diagnostic used `30` GPU wall seconds against the `1800` second hard limit; the guard rejected the first optimizer step and retained the baseline.
- The precision audit used `0` additional GPU seconds. The total GPU wall time for the E019 diagnostic was `30` seconds; no formal RL, C-Eval, frozen-set evaluation, or Gemini call was run.
- The wrapper recorded commit `442da86bce698b64f15f080a6e9109df17a17ff7`, environment hash, command, resource monitor, exit code, and wall time. The GPU returned to `0 MiB`, disk remained approximately `92 GiB` available, and the server stayed `RUNNING`.
- No USD billing export was available; no dollar amount is inferred from the credit balance.

## 2026-08-02 MM-E018 / MM-F026 telemetry smoke

- The single authorized GRPO smoke used `13` GPU wall seconds against the `1800` second hard limit and exited with code `1` before the first optimizer attempt was recorded.
- The failure was a scalar AdamW `step` tensor digest edge case. The run log and all artifacts produced before failure were preserved; no checkpoint or output weight was created. The scalar-hash fix and regression tests ran offline afterward; no second GPU run was started.
- The incomplete audit used `0` GPU seconds. After completion the L4 was at `0 MiB`, disk availability was approximately `92 GiB`, and the server remained `RUNNING`.
- No C-Eval, frozen-set evaluation, Gemini call, or USD billing export was used; no dollar amount is inferred.

## 2026-08-02 MM-E017 / MM-F025 KL guard diagnostic

- The isolated run root was `results/experiments/rl_kl_guard_diagnostic_20260802/`; it used one L4 and one GRPO seed. Two smoke attempts used `11 + 11` GPU wall seconds and the formal run used `27` seconds, for `49` seconds total against the `3600` second hard limit.
- The wrapper preserved command, training commit `009b1eece9cd7ceece08a3e5f76b3f47b42e6892`, implementation archive commit `26b78fc645953192de55317383c580a2269f5486`, environment hash, GPU/disk snapshots, exit code, resource monitor, and wall time. The offline audit added no GPU work.
- The guard rejected the first formal optimizer step after four attempts and restored the policy/AdamW state; no new weight or checkpoint was created. Baseline was retained.
- After completion the GPU was at `0 MiB`, disk availability was approximately `92 GiB`, and the server remained `RUNNING`. No C-Eval, frozen-set evaluation, Gemini call, or USD billing export was used, so no dollar amount is inferred.

## 2026-08-02 MM-E018 / MM-F026 corrected retry

- The explicitly authorized corrected GRPO smoke used `30` GPU wall seconds against the `1800` second hard limit and exited normally after the KL guard rejected the first optimizer step.
- The first failed attempt used `13` seconds, so the two E018 smoke attempts consumed `43` GPU wall seconds total. The retry audit used `0` additional GPU seconds.
- The wrapper recorded command, commit `86e65c4bd37410abdc90de6de31dc8e8f396f4dc`, environment hash, GPU/disk snapshots, exit code, resource monitor, and wall time in the separate retry root. The GPU returned to `0 MiB`, disk remained approximately `92 GiB` available, and the server stayed `RUNNING`.
- No C-Eval, frozen-set evaluation, Gemini call, or USD billing export was used; no dollar amount is inferred from the credit balance.

## 2026-08-02 MM-E021 / MM-F029 corrected KL token replay smoke

- The single GRPO smoke used `30` GPU wall seconds against an `1800` second hard limit and exited with code `0`; four post-step guard attempts were recorded, zero optimizer steps were accepted, and zero checkpoints were retained.
- The offline replay audit used `0` GPU seconds. It validated 24 token-replay rows and exact rollback evidence. No formal RL, CISPO, C-Eval, frozen-set evaluation, or Gemini call was run.
- Before and after snapshots showed the L4 idle at `0 MiB` and approximately `92 GiB` available disk; the server remained `RUNNING`. No Cloud Billing export was available, so no USD amount is inferred from the credit balance.

## 2026-08-02 MM-E022 / MM-F030 corrected KL gate smoke

- The single authorized GRPO smoke completed 2/2 steps in `26` GPU wall seconds against the `1800` second hard limit; no retry, formal run, CISPO, C-Eval, frozen-set evaluation, or Gemini call was started.
- The wrapper recorded commit `7ef8a0f4fb103ef255762e6d461c7df84ddbad7e`, environment hash `4f276ccd1cf9cbced848ee23b7cea5da3dff480eda3b276bb5928afcd9ae5e3d`, the full command, resource snapshots, exit code `0`, and wall time. The offline audit used `0` additional GPU seconds.
- One diagnostic step-2 checkpoint was retained and reloaded; it is not the default model and is not a promoted model artifact.
- After completion the server remained `RUNNING`, the L4 returned to `0 MiB`, and approximately `92 GiB` disk remained available. No Cloud Billing export was available, so no USD amount is inferred.

## 2026-08-02 MM-E023 / MM-F031 pre-step precision smoke

- The single authorized GRPO smoke completed 2/2 steps in `25` GPU wall seconds against the `1800` second hard limit. The offline audit added `0` GPU seconds.
- Commit `5fbee7b84a08317002ba5291ec10326c443594bc` and environment hash `83e8de54c41d026d2b0dc81b249f26e225c95e42b25a3ba8a651909a92407d40` are recorded with the command, resource snapshots, exit code `0`, and wall time.
- No retry, formal run, CISPO, C-Eval, frozen-set evaluation, Gemini call, optimizer change, or model promotion was performed. The diagnostic checkpoint is retained only for reload evidence.
- The server remains `RUNNING`, the L4 is idle at `0 MiB`, and approximately `91 GiB` disk is available. No Cloud Billing export was available, so no USD amount is inferred.
## 2026-08-02 MM-E024 / MM-F032 FP32 training-forward smoke

- GPU task: one GRPO seed-42 smoke; no formal run, CISPO, C-Eval, frozen evaluation, or Gemini call.
- Hard wall limit: 1800 seconds; observed GPU wall time: 26 seconds.
- GPU: NVIDIA L4; memory after run: 0 MiB.
- Disk after run: approximately 91 GiB available.
- Server state after run: `RUNNING`.
- Output root: `results/experiments/rl_fp32_training_forward_smoke_20260802/`.
- Result: diagnostic-only; no default weight, reward, checkpoint-selection rule, or promotion gate changed.

## 2026-08-02 MM-E025 / MM-F033 nonzero-advantage contract

- The contract wrapper and audit were offline-only with `CUDA_VISIBLE_DEVICES` disabled; GPU wall time was `0` seconds and total wall time was approximately `2` seconds.
- The wrapper recorded commit `ba52712c80fc5038eb8daa92baf20b1681f6bfed`, environment hash `835f30224da01815963da9ee8e093ae03847f27da8189351c8195c7cbb44a402`, command, exit code `0`, disk snapshots, and the audit status.
- No model was loaded, no checkpoint or weight was written, and no Gemini, C-Eval, frozen-set, or formal RL task was run. No USD amount is inferred because no Cloud Billing export was used.
- The server remains `RUNNING`; the L4 is idle at `0 MiB` and approximately `91 GiB` disk is available.

## 2026-08-02 MM-E026 / MM-F034 controlled active-path smoke

- One GRPO seed-42 smoke ran under the `1800` second hard limit; observed GPU wall time was `26` seconds. No formal RL, CISPO, multi-seed, C-Eval, frozen-set evaluation, or Gemini call was started.
- The wrapper recorded commit `6406cd13e96a06e9d3f8835c598ba133cce0ad78`, environment hash `2ba7e184507d70758d483048e20586628daee0fb2f691f3780dde152ee06c8d9`, command, resources, exit code `0`, and wall time. The corrected offline audit used zero additional GPU seconds.
- The server remained `RUNNING`, the L4 returned to `0 MiB`, and approximately `91 GiB` disk remained available. No USD amount is inferred because no Cloud Billing export was used.

## 2026-08-02 MM-E027 / MM-F035 natural rule-reward smoke

- One GRPO seed-42 smoke ran under the `1800` second hard limit; observed GPU wall time was `26` seconds. No formal RL, CISPO, multi-seed, C-Eval, frozen-set evaluation, or Gemini call was started.
- The wrapper recorded commit `8cf5f6971e29e0ba913aee729ae6e946a89cb392`, environment hash `4711e5e81656d6290bb1bb0248c3bfc46a33015551751b0a39c1c1ca16cd9a21`, command, resource snapshots, exit code `0`, and wall time. The corrected offline audit used zero additional GPU seconds.
- The server remained `RUNNING`, the L4 returned to `0 MiB`, and approximately `91 GiB` disk remained available. No USD amount is inferred because no Cloud Billing export was used.

## 2026-08-02 MM-E028 / MM-F036 natural reward diversity audit

- This was an offline artifact audit only. The wrapper disabled CUDA, scanned 13 sample files, and used `0` GPU wall seconds; no formal RL, CISPO, benchmark, or Gemini call ran.
- The wrapper recorded the command, commit, environment hash, resource snapshots, exit code `0`, and audit output under `results/experiments/rl_natural_reward_diversity_audit_20260802/`.
- The server remained `RUNNING`; the L4 was idle at `0 MiB`; approximately `91 GiB` remained available. No USD amount is inferred without a Cloud Billing export.

## 2026-08-02 MM-E029 / MM-F037 natural reward coverage audit

- This was an offline audit of one manifest and one persisted sample root. It disabled CUDA, used `0` GPU wall seconds, and did not run model inference, RL, CISPO, C-Eval, frozen evaluation, or Gemini.
- The wrapper recorded command, commit, environment hash, GPU/disk snapshots, exit code `0`, and audit output under `results/experiments/rl_natural_reward_coverage_audit_20260802/`. A wrapper hash-path warning from the first run was preserved; the corrected wrapper was validated separately in a temporary output root without GPU work.
- The server remained `RUNNING`; the L4 was idle at `0 MiB`; approximately `91 GiB` remained available. No USD amount is inferred without a Cloud Billing export.
## 2026-08-02 MM-E030 / MM-F038 balanced coverage smoke

- One GRPO seed-42 smoke ran for 2 steps with eight balanced categories. GPU wall time was 33 seconds against the 1800-second hard limit; no formal RL, CISPO, multi-seed, C-Eval, frozen-set evaluation or Gemini call ran.
- The wrapper recorded command, commit `54c7621a7bcd6cf20cf46344e7a3fa2a0c946c61`, environment hash, GPU/disk snapshots, exit code `0`, checkpoint and offline coverage audit. The audit used zero additional GPU time.
- The server remains `RUNNING`, the L4 is idle at `0 MiB`, and approximately `90 GiB` remains available. No USD amount is inferred without a Cloud Billing export. No default model or existing experiment root was changed.
## 2026-08-02 MM-E031 / MM-F039 balanced output-quality audit

- This was a CUDA-disabled offline replay of E030. GPU wall time was `0` seconds; no model inference, formal RL, CISPO, C-Eval, frozen evaluation or Gemini call ran.
- The wrapper recorded command, commit `4167a3dedff1c03d794402e04604e2eabe5cbcb5`, environment hash, resource snapshots, exit code `0` and an isolated audit root. No checkpoint or weight was written.
- The server remains `RUNNING`, the L4 is idle at `0 MiB`, and approximately `90 GiB` remains available. No USD amount is inferred without a Cloud Billing export.
## 2026-08-02 MM-E032 / MM-E033 / MM-F040 / MM-F041 release gate

- Preflight GRPO seed 42: `121` GPU wall seconds.
- Six formal GRPO/CISPO runs: `3964` GPU wall seconds total (`660–661` seconds per run); all exit codes were `0`.
- Frozen evaluation of baseline plus six selected checkpoints: `678` GPU wall seconds total; all seven evaluations exited `0`.
- Total recorded GPU wall time: `4763/14400` seconds, below the four-hour hard limit. The budget state is preserved at `results/experiments/rl_release_gate_20260802_retry1/budget_state.json`.
- The initial requested-root wrapper failure exited `127` before GPU work and used `0` GPU wall seconds; its log is preserved separately and was not overwritten.
- Final resources: L4 `0 MiB`, approximately `83 GiB` disk available, server `RUNNING`. No C-Eval or Gemini call ran in this stage. No USD amount is inferred without a Cloud Billing export.
## 2026-08-03 MM-E034 / MM-F042 quality-repair cost

- Input audit and fail-closed selection only; GPU wall time: `0/14400` seconds.
- Baseline validation, SFT smoke and candidate validation: not run.
- Final resources: L4 idle at `0 MiB`, approximately `83 GiB` disk available, server `RUNNING`.
- No model weight or default model was changed. No USD cost is inferred without a Cloud Billing export.
## 2026-08-03 MM-E034 / MM-F042 authorized supplement and SFT smoke

- Supplement preparation and input/selection audit: GPU wall time `0` seconds.
- Baseline validation: `71` seconds; SFT repair: `34` seconds; candidate validation: `68` seconds; total recorded GPU wall time `173/14400` seconds.
- Final resources: L4 idle at `0 MiB`, approximately `82 GiB` disk available, server `RUNNING`.
- The candidate checkpoint is retained only in its isolated experiment directory. The default model was not changed. No USD amount is inferred without a Cloud Billing export.
## 2026-08-03 MM-E035 / MM-F043 corrected-GRPO smoke

- One GRPO seed-42 smoke ran for 2 optimizer steps; recorded GPU wall time was `61/1800` seconds.
- No backoff or rejected update occurred. The step-2 checkpoint and audit were written in a new root; no formal RL, CISPO, C-Eval, Gemini or frozen evaluation ran.
- Final resources: L4 `0 MiB`, approximately `81 GiB` disk available, server `RUNNING`.
- No USD amount is inferred without a Cloud Billing export; the candidate and default weights remain separate.
## 2026-08-03 MM-F044 precision audit

- Offline CUDA-disabled replay/audit only; GPU wall time: `0` seconds.
- No new checkpoint, training task, benchmark, Gemini call or model change occurred.
- Final resources remained L4 `0 MiB`, approximately `81 GiB` disk available, server `RUNNING`.

## 2026-08-03 MM-E036 / MM-F045 precision contract smoke

- Static implementation checks and 46 related regression tests used `0` GPU wall seconds.
- One GRPO seed-42 smoke completed 2 optimizer steps in `26/1800` GPU wall seconds; no formal RL, CISPO, C-Eval, frozen evaluation or Gemini call ran.
- Audit and checkpoint reload used no additional GPU time. Final L4 usage was `0 MiB`, approximately `81 GiB` remained available, and the server stayed `RUNNING`.
- The output checkpoint remains isolated under `results/experiments/rl_precision_contract_smoke_20260803/`; no USD amount is inferred without a Cloud Billing export and no default weight was changed.
## 2026-08-03 MM-E037 / MM-F046 four-step precision-contract diagnostic

- Static checks, wrapper correction and offline audit used no GPU wall time beyond the recorded run.
- One GRPO seed-42 run completed 4 optimizer steps in `45/1800` GPU seconds; the wrapper elapsed time was `60` seconds because of the 60-second resource-monitor interval.
- The first automatic audit exited `1` due only to the wrapper pre-creating the non-empty audit directory. The failure log was retained, the wrapper was corrected, and the independent final audit exited `0` without rerunning GPU.
- Final resources: L4 `0 MiB`, approximately `81 GiB` disk available, server `RUNNING`. No USD amount is inferred without a Cloud Billing export; no default weight was changed.
## 2026-08-03 MM-E038 / MM-F047 offline precision/quality audit

- CUDA-disabled JSON/JSONL audit only; GPU wall time: `0` seconds.
- No model load, checkpoint write, benchmark, Gemini call or training task occurred.
- Final L4 usage was `0 MiB`, approximately `81 GiB` disk available, server `RUNNING`.
- No USD amount is inferred without a Cloud Billing export; no default weight changed.
## 2026-08-03 MM-E039 / MM-F048 quality evidence boundary

- Pure offline JSON/JSONL audit; no model load, checkpoint write or GPU work. GPU wall time: `0` seconds.
- Final resources: L4 `0 MiB`, approximately `81 GiB` disk available, server `RUNNING`.
- No Gemini call or benchmark ran. No USD amount is inferred without a Cloud Billing export; no default weight changed.
## 2026-08-03 MM-E040 / MM-F049 corrected-GRPO quality evidence diagnostic

- One GRPO seed-42 run completed four steps; recorded training GPU wall time was `97/3600` seconds and wrapper wall time was `120` seconds.
- Precision and quality audits used no additional GPU work. Final L4 usage was `0 MiB`, approximately `80 GiB` disk available, server `RUNNING`.
- No CISPO, multi-seed matrix, C-Eval, frozen evaluation or Gemini call ran. No USD amount is inferred without a Cloud Billing export; no default weight changed.
## 2026-08-03 MM-E041 / MM-F050 failure attribution audit

- Pure CUDA-disabled JSON/JSONL audit; no model load, checkpoint write or GPU training. GPU wall time: `0` seconds.
- The audit processed 256 generated samples and 32 validation records. Existing incomplete audit directories were preserved; the final result used a fresh `quality_failure_attribution_audit_v3` directory.
- Final resources: L4 `0 MiB`, approximately `80 GiB` disk available, server `RUNNING`.
- No Gemini, benchmark or formal RL run occurred. No USD amount is inferred without a Cloud Billing export; no default weight changed.
## 2026-08-03 MM-E042 / MM-F051 reward-input contract audit

- CUDA-disabled offline audit only; no model load, checkpoint write or training. GPU wall time: `0` seconds.
- Replayed 128 chosen records and 256 generated samples; no GPU, Gemini, benchmark or formal RL work occurred.
- Final resources: L4 `0 MiB`, approximately `80 GiB` disk available, server `RUNNING`.
- No USD amount is inferred without a Cloud Billing export; no default weight changed.
## 2026-08-03 MM-E043 / MM-F052 Output-to-Validator mapping audit

- CUDA-disabled artifact audit only; no model load, checkpoint write or training. GPU wall time: `0` seconds.
- Audited 128 manifest rows and 256 generated samples; wrote 113 linked failure cases to a fresh directory.
- Final resources: L4 `0 MiB`, approximately `80 GiB` disk available, server `RUNNING`.
- No Gemini, benchmark or formal RL work occurred. No USD amount is inferred without a Cloud Billing export; no default weight changed.

## MM-E044 / MM-F053 category-weighting audit

- Execution: CUDA-disabled offline JSON/JSONL audit; no model load and no GPU task.
- GPU wall time: `0` seconds; L4 after audit: `0 MiB`.
- Disk after audit: approximately `80 GiB` available.
- Server state: `RUNNING`.
- Output root: `results/experiments/rl_category_weighting_audit_20260803_v2/`.
- Existing v1 audit root and all prior experiment roots were preserved.
- No new cloud training cost was incurred; billing was not inferred beyond the recorded zero GPU wall time.

## 2026-08-03 MM-E045 / MM-F054 preference repair smoke

- One fresh retry completed SFT, DPO, SimPO and common 160 plus 32 evaluation; the earlier checkpoint-save failure and logs were preserved.
- Recorded phase wall time: 348 seconds; final L4 0 MiB; disk about 78.7 GiB; server RUNNING.
- No Gemini, C-Eval, frozen evaluation, formal RL or default-weight replacement occurred; no USD amount is inferred without a billing export.


## MM-E047 / MM-F056：偏好修复后 corrected-GRPO 成本

- corrected-GRPO GPU wall time：约 `97` 秒；完成后 L4 为 0 MiB，服务器保持 `RUNNING`。
- 本轮未运行 CISPO、三 seed、C-Eval 或冻结集；无新增外部评分器调用。
- 结果目录独立保存，默认权重和既有实验目录未覆盖。
