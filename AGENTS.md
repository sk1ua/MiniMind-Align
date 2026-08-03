# MiniMind-Align 工作约定

MiniMind-Align 使用约 64M 参数 MiniMind decoder-only Transformer 建立低资源、可复现、可验证的指令对齐、偏好优化、Reward Model、GRPO/CISPO 与对齐税评测平台。当前 Sprint 为 A（Alignment v2 数据工程），先完成现场核验、实验冻结、smoke、pilot 和 fail-closed 审计。

远端项目为 /home/sakuaikacn/minimind；服务器为 NVIDIA L4 24GB；MiniMind 环境为 .venv，教师/评审环境为 .venv-teacher；模型配置为 hidden_size=768、num_hidden_layers=8、约 63.91M 参数，训练优先使用 bfloat16。

## 不可覆盖与数据泄漏

- 不覆盖或删除已有 out/、checkpoints/、results/、dataset/alignment_v1/、日志、判断、权重或用户未提交修改。
- 新实验使用唯一 experiment id、权重名和输出目录；目录已存在时必须失败。
- 不提交大权重、API key、访问令牌、服务账户密钥或 Gemini 原始敏感响应。
- 冻结 evaluation/data/ 与 dataset/alignment_v1/splits/prompts_test.jsonl 为独立评测来源。
- 训练/验证禁止包含测试 prompt、测试答案、人工复核回答或 test_generation 抽出的 chosen。
- 审计必须检查精确匹配、Unicode 规范化、去标点/空白/数字归一化、n-gram/Jaccard、SequenceMatcher 和模板/家族相似度；任一关键项失败即退出非零。
- train 与 validation 按 prompt 隔离；所有数据保存 SHA256。

## 训练、审计和代码规则

- 所有随机过程固定 seed；记录配置、命令、环境、日志、结果、权重、时长、显存、token 和成本。
- 每次 GPU 任务前检查 nvidia-smi、磁盘、内存、训练进程和 tmux；同一时间只运行一个 GPU 训练任务；长任务在 tmux 中运行。
- 每个完整训练先执行 10--20 step smoke，检查有限 loss、无 NaN、保存/回载和推理；失败保留日志，同一错误最多修复三次后标记 BLOCKED。
- Python 使用类型注解、pathlib、UTF-8 JSONL、argparse、docstring；训练和推理 tokenizer/chat template 一致；明确 prompt/response mask；不记录 chain-of-thought。

## 执行与汇报

- 不创建或切换 Git 分支，不重置仓库，不提交用户未提交修改；每个 Sprint 独立 commit。
- 磁盘少于 30GB、单次训练超过 6 GPU 小时、剩余任务超过 30 GPU 小时、严重测试泄漏、NaN/权重损坏或未估计的额外云账单时暂停。
- 汇报必须给出真实文件路径、命令、数量、指标、日志、验收、阻塞、GPU 时间和成本，并区分事实、推断、假设、失败和未完成。
