# Chem Agent · 化学工具助手

![Chem Agent 项目头图 / Project header](docs/images/header.png)

用 DeepSeek 选择工具、RDKit 计算分子性质与结构警示的中文化学助手，附带 v1（40 题）和 v2（60 题）评测及真实迭代记录。


[English](#english)

## 功能

| RDKit 工具 | 功能 |
| --- | --- |
| `mol_weight(smiles)` | 平均分子量，保留两位小数；数值对应摩尔质量 g/mol |
| `logp(smiles)` | Wildman–Crippen 预测 LogP |
| `formula(smiles)` | 分子式 |
| `tpsa(smiles)` | 拓扑极性表面积，单位 Å² |
| `hbd_hba(smiles)` | 氢键供体数和受体数 |
| `lipinski(smiles)` | Lipinski 指标、违反项数和筛选结果；当前实现允许最多违反一项 |
| `similarity(smiles_a, smiles_b)` | Morgan 指纹的 Tanimoto 相似度，半径 2、2048 位，不编码手性 |
| `safety_check(smiles)` | 使用 RDKit FilterCatalog 内置 Brenk（BRENK）与 NIH 规则，返回结构警示名称、目录、论文来源及免责声明 |

- **Function-calling Agent**：使用 OpenAI 兼容 SDK 手写循环，不依赖 LangChain；模型选工具，本地执行并回传结果，最终用中文回答。默认最多请求模型 5 次，保留每次工具调用的参数和结果。
- **两版题集**：v1 为 40 题；v2 保留这些题，新增安全检查 8 题、多步骤 6 题、易混淆 6 题，共 60 题。期望值由本地工具生成，支持有序多工具评分、按类别汇总、错题记录与 token 统计。
- **安全规则有出处**：只使用 RDKit 内置目录，不手写 SMARTS。Brenk et al., *ChemMedChem* (2008)，DOI [10.1002/cmdc.200700139](https://doi.org/10.1002/cmdc.200700139)；Jadhav et al., *J Med Chem* (2010)，DOI [10.1021/jm901070c](https://doi.org/10.1021/jm901070c)。完整论文名称见文末“规则来源”。
- **输入检查**：工具遇到非法 SMILES 返回 `error`；单道评测题异常记为失败并继续。
- **三题演示**：依次展示分子量、分子式和相似度问题的工具名称、参数、计算结果及最终回答。

**安全免责声明：仅基于结构的规则粗筛（药物化学警示库），不是毒性或爆炸性评估，请以 MSDS/SDS 为准。** 无警示不等于安全。Agent 的 system prompt 要求转述免责声明，且**不提供合成路线或制备方法**；当前评测不能证明所有真实请求都能正确遵从这些要求。

### 氢键计数口径

`hbd_hba` 和 `lipinski` 均使用 RDKit 的 `Lipinski.NumHDonors` / `Lipinski.NumHAcceptors`，不是简单统计所有 N、O 原子。已保存的咖啡因结果为 **HBA=3**（q022、q028）；其分子式 **C8H10N4O2**（q012）对应 **N+O=6**，这是 Lipinski 原文所用的受体近似计数口径。两个数字定义不同，不能互换。本项目的 `lipinski` 使用前一种 RDKit 氢键定义。

依据：[咖啡因分子式记录](eval/results/20261018_221950_033160.jsonl#L12)、[HBD/HBA 记录](eval/results/20261018_221950_033160.jsonl#L22)、[Lipinski 记录](eval/results/20261018_221950_033160.jsonl#L28)及 [RDKit 对两种计数的说明](https://www.rdkit.org/docs/cppapi/Lipinski_8h.html)。

## 架构

**问题 → DeepSeek 选工具 → RDKit 本地计算 → 工具结果回传 DeepSeek → 中文回答**

![Chem Agent 架构图 / Architecture](docs/images/architecture.png)

`chem_agent/tools.py` 提供函数注册表和 JSON Schema；`chem_agent/agent.py` 负责模型请求与工具循环；`eval/run_eval.py` 逐题运行并与期望值比较。

当前模型为 **`deepseek-flash` 非思考模式**，接口为 `https://api.deepseek.com`。密钥仅从环境变量 `LLM_API_KEY` 读取；本地 RDKit 工具和离线测试无需密钥。历史 v1 评测使用 `deepseek-chat`，不能用当前模型重跑后的结果替代历史记录。

## 真实评测结果

### v1：40 题，deepseek-chat

以下数字来自 [v1 真实评测报告](eval/results/summary.md)，运行时间为 **2026-10-18 22:19:50–22:21:17（UTC+8）**，模型为 `deepseek-chat`。

| 指标 | 真实结果 |
| --- | ---: |
| 题目总数 | 40 |
| 总准确率 | 100.00%（40/40） |
| 工具选择准确率 | 100.00%（40/40） |
| 数值/结果准确率 | 100.00%（40/40） |
| 异常数 | 0 |
| 错题数 | 0 |

各类别均全部通过；可查看 [40 题原始记录](eval/results/20261018_221950_033160.jsonl)。

### v2：60 题，deepseek-flash 非思考模式

**v1 与 v2 的题集和模型均不同，分数不能直接对比。** 以下仅比较同一 v2 题集、相同模型设置下的改动前后两轮；数据来自 [2026-10-19 前后对比表](eval/results/v2_comparison.md)及其链接的原始结果。

| 指标 | 改进前 | 改进后 |
| --- | ---: | ---: |
| 总准确率 | 100.00%（60/60） | 100.00%（60/60） |
| 工具选择准确率 | 100.00%（60/60） | 100.00%（60/60） |
| 输入 token | 167,918 | 172,397 |
| 输出 token | 11,176 | 10,399 |
| 总 token | 179,094 | 182,796 |

两轮之间只在 system prompt 末尾追加一句要求：**仅检查 SMILES 有效性时，最多调用一个工具一次；确认无效后不重复验证，不猜测或举例补全、修改结构。** q038、q039 的工具调用各从 **2 次降至 1 次**，回答不再包含此前的结构补全或修改示例。这是本次观察到的行为改进；**准确率持平，总 token 增加 3,702，并非准确率提升或整体 token 节省**。完整分析及对应原始行号见 [v2_error_analysis.md](eval/results/v2_error_analysis.md)。

### 失败案例与局限

- 自动评分没有错题，说明题集在当前模型与评分口径下仍偏简单、区分度有限，不能据此推断真实化学问题上的普遍可靠性。
- q038/q039 的冗余调用和结构举例未被自动评分判错，依靠人工复核才发现；q056 也从 2 次调用降至 1 次，但不是提示词针对的题，可能是采样波动。
- 单工具题只检查第一条调用及其结果；多工具题检查完整有序调用序列和逐项结果。浮点数按绝对容差比较，整数、字符串、布尔值精确匹配。非法题检查回答是否明确说明输入无效，总通过还要求取得最终回答。
- **评分不检查自然语言解释里的所有数值或科学表述**；工具结果正确不保证解释完全正确。两种提示词各只运行一轮，且在同一题集上诊断、复测，不构成泛化提升或因果效果的证据。

## 快速开始

以下命令在项目根目录的 **Windows PowerShell** 中执行；本机验证环境为 **Python 3.12.5**。

### 1. 建立虚拟环境并安装依赖

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

已有 `.venv` 时跳过创建；以后每次打开终端，先执行 `.\.venv\Scripts\Activate.ps1`。`requirements.txt` 锁定当前环境的直接依赖及间接依赖版本。

### 2. 设置密钥并运行示例

```powershell
$env:LLM_API_KEY = "替换为你的 DeepSeek API key"
python -m chem_agent.agent "阿司匹林的分子量是多少"
python examples/demo.py
```

密钥设置只影响当前 PowerShell 会话；不要把实际密钥写入源码。若尚无密钥，可先运行纯本地示例：

```powershell
python examples/min_tool_demo.py
```

`demo.py` 在每道问题完成后，按顺序展示该题记录的每一步工具调用和最终回答。

### 3. 跑测试与评测

```powershell
# 以下命令不需要密钥或网络
python -m pytest -q
python eval/build_questions.py --check
python eval/run_eval.py --dry-run --limit 3
python eval/run_eval.py --questions eval/questions_v2.jsonl --dry-run

# 以下命令使用真实 API，需要 LLM_API_KEY
python eval/run_eval.py --limit 5 --output-dir eval/results/local_v1
python eval/run_eval.py --questions eval/questions_v2.jsonl --output-dir eval/results/local_v2
```

默认题集仍为 v1；`--questions` 可指定 v2，`--limit N` 只跑前 N 题。上面的真实命令显式指定新的输出目录，避免覆盖仓库保留的汇总；每次写入带时间戳的 JSONL、摘要及该目录的 `summary.md`。`--dry-run` 将报告放在对应输出目录的 `dry_run/` 中，只用期望值构造 mock，模拟成绩不代表模型能力。`build_questions.py --check` 同时校验两版；不带此参数时生成 v2，保留 v1。

## 项目结构

```text
chem-agent/
├── chem_agent/
│   ├── __init__.py
│   ├── tools.py            # RDKit 工具与函数描述
│   └── agent.py            # DeepSeek function-calling 循环
├── examples/
│   ├── min_tool_demo.py    # 无需密钥的本地示例
│   └── demo.py             # 三种问题的完整 Agent 演示
├── eval/
│   ├── build_questions.py  # 生成、校验期望值
│   ├── questions.jsonl     # v1：40 道中文题
│   ├── questions_v2.jsonl  # v2：60 道中文题
│   ├── run_eval.py         # 自动评分、token 统计
│   ├── compare_results.py  # 从保存的结果生成前后对比
│   └── results/            # 逐题详情和汇总报告
├── tests/                 # 工具、Agent、题集与评分器测试
├── docs/
│   └── images/            # 头图与架构图
├── requirements.txt       # 当前环境的版本锁定
├── LICENSE                # MIT，作者 czhou143
└── README.md
```

## 后续计划

- [ ] MCP 封装：将现有工具暴露为 MCP 服务，供支持 MCP 的客户端调用。

MCP 封装尚未实现；结构警示工具与项目图片已经提供。项目采用 [MIT License](LICENSE)，Copyright (c) 2026 czhou143。

---

## English

A chemistry assistant that uses DeepSeek to select tools and RDKit to compute molecular properties and structural alerts, with v1 (40-question) and v2 (60-question) evaluations and saved iteration results.

### Features

| RDKit tool | Purpose |
| --- | --- |
| `mol_weight` | Average molecular weight, rounded to two decimals |
| `logp` | Wildman–Crippen predicted LogP |
| `formula` | Molecular formula |
| `tpsa` | Topological polar surface area in Å² |
| `hbd_hba` | Hydrogen-bond donor and acceptor counts |
| `lipinski` | Lipinski values, violation count and screening result; at most one violation passes |
| `similarity` | Morgan-fingerprint Tanimoto similarity; radius 2, 2048 bits, without chirality |
| `safety_check` | RDKit FilterCatalog's built-in Brenk (BRENK) and NIH rules; returns alert names, catalogs, references and a disclaimer |

- A handwritten function-calling loop using the OpenAI-compatible SDK, without LangChain. It defaults to five model requests, records tool arguments/results, and answers in Chinese.
- Two datasets: v1 contains 40 questions; v2 retains them and adds 8 safety, 6 multi-tool and 6 easily confused questions, for 60 in total. Expected values come from local tools; the runner supports ordered tool calls, category reports and token accounting.
- Source-backed screening using built-in catalogs, without handwritten SMARTS: Brenk et al., *ChemMedChem* (2008), DOI [10.1002/cmdc.200700139](https://doi.org/10.1002/cmdc.200700139); Jadhav et al., *J Med Chem* (2010), DOI [10.1021/jm901070c](https://doi.org/10.1021/jm901070c). Full paper titles are listed under References below.
- Invalid SMILES return an `error`; individual evaluation exceptions are recorded as failures without stopping the batch.
- A three-question demo prints each recorded tool call and the final answer after each question finishes.

**Disclaimer: this is a structure-based preliminary screen using medicinal-chemistry alert catalogs, not a toxicity or explosivity assessment; consult the MSDS/SDS.** No alerts do not establish safety. The system prompt requires the disclaimer and prohibits **synthesis routes or preparation methods**; the current evaluation does not establish compliance for every real-world request.

#### Hydrogen-bond definitions

Both `hbd_hba` and `lipinski` use RDKit's `Lipinski.NumHDonors` / `Lipinski.NumHAcceptors`. Saved caffeine results give **HBA=3** (q022/q028). Its saved formula **C8H10N4O2** (q012) gives **N+O=6**, corresponding to the original Lipinski acceptor-count approximation. These are different definitions, not interchangeable values; this project's screening uses the RDKit hydrogen-bond definitions.

Sources: saved [formula](eval/results/20261018_221950_033160.jsonl#L12), [HBD/HBA](eval/results/20261018_221950_033160.jsonl#L22), [Lipinski result](eval/results/20261018_221950_033160.jsonl#L28), and [RDKit's descriptor definitions](https://www.rdkit.org/docs/cppapi/Lipinski_8h.html).

### Architecture

**Question → DeepSeek selects a tool → RDKit computes locally → Results return to DeepSeek → Final answer**

`chem_agent/tools.py` defines tools and JSON schemas, `chem_agent/agent.py` runs the model/tool loop, and `eval/run_eval.py` compares results with expected values. The current model is **`deepseek-flash` in non-thinking mode** at `https://api.deepseek.com`; credentials come from `LLM_API_KEY`. The historical v1 evaluation used `deepseek-chat`; rerunning it with the current model does not reproduce that historical configuration.


### Live evaluation results

#### v1: 40 questions, deepseek-chat

Source: [saved v1 report](eval/results/summary.md), **2026-10-18 22:19:50–22:21:17 (UTC+8)**, using `deepseek-chat`.

| Metric | Live result |
| --- | ---: |
| Questions | 40 |
| Overall accuracy | 100.00% (40/40) |
| Tool selection accuracy | 100.00% (40/40) |
| Value/result accuracy | 100.00% (40/40) |
| Exceptions | 0 |
| Failed questions | 0 |

Every category passed; [raw records](eval/results/20261018_221950_033160.jsonl) are available.

#### v2: 60 questions, deepseek-flash in non-thinking mode

**Both the dataset and model differ between v1 and v2, so their scores are not directly comparable.** The following compares two runs on the same v2 dataset with identical model settings. Values come from the [2026-10-19 comparison report](eval/results/v2_comparison.md) and its raw result files.

| Metric | Before | After |
| --- | ---: | ---: |
| Overall accuracy | 100.00% (60/60) | 100.00% (60/60) |
| Tool selection accuracy | 100.00% (60/60) | 100.00% (60/60) |
| Prompt tokens | 167,918 | 172,397 |
| Completion tokens | 11,176 | 10,399 |
| Total tokens | 179,094 | 182,796 |

The only experimental change was one instruction appended to the system prompt: use at most one tool once for a SMILES-validity check; after confirming invalid input, do not repeat validation or speculate about completed or modified structures. Tool calls for q038 and q039 each decreased from **2 to 1**, and their answers stopped including the earlier structure-repair examples. This is an observed behavioral improvement: **accuracy was unchanged and total tokens increased by 3,702**, not an accuracy gain or an overall token saving. See the [analysis and original line references](eval/results/v2_error_analysis.md).

#### Failure cases and limitations

- There were no automatically scored failures. This suggests the dataset remains relatively easy and has limited discriminatory power for this model/rubric; full marks do not establish general reliability on chemistry tasks.
- The rubric passed q038/q039 despite redundant tool calls and structure-repair examples; these were found through manual review. q056 also went from 2 calls to 1, but was not targeted by the change and may reflect sampling variation.
- Single-tool questions score the first call and its result; multi-tool questions require the full ordered sequence and corresponding results. Floats use absolute tolerances; integers, strings and booleans match exactly. Invalid-input questions require an explicit invalid-input statement, and overall success requires a final answer.
- **Scoring does not check every number or scientific claim in the natural-language explanation.** Correct tool results do not guarantee a correct explanation. Each prompt setting was run once, with diagnosis and retesting on the same dataset; this does not establish generalization or causality.

### Quick start

From the project root in **Windows PowerShell**, with **Python 3.12** (tested on 3.12.5):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:LLM_API_KEY = "<your DeepSeek API key>"
python -m chem_agent.agent "阿司匹林的分子量是多少"
python examples/demo.py

# Offline: no API key or network required
python examples/min_tool_demo.py
python -m pytest -q
python eval/build_questions.py --check
python eval/run_eval.py --dry-run --limit 3
python eval/run_eval.py --questions eval/questions_v2.jsonl --dry-run

# Live: requires an API key
python eval/run_eval.py --limit 5 --output-dir eval/results/local_v1
python eval/run_eval.py --questions eval/questions_v2.jsonl --output-dir eval/results/local_v2
```

Skip environment creation if `.venv` exists; activate it in each new terminal. Requirements pin the installed direct and transitive dependencies. Keep real credentials out of source files; the command above sets the key only for the current session.

The default dataset remains v1; select v2 with `--questions`, or use `--limit N` for a prefix. The live examples explicitly choose new output directories to preserve the archived summaries. Each run writes timestamped JSONL details and a summary, plus that directory's latest `summary.md`. Dry runs save under the corresponding `dry_run/` directory and use expected answers as mock responses; their scores do not measure model quality. The generator's `--check` validates both datasets; generation without it writes v2 while preserving v1.

### Project layout

- `chem_agent/`: RDKit tools, schemas and the Agent loop.
- `examples/`: a local RDKit example and the three-question Agent demo.
- `eval/`: v1/v2 datasets, generation/validation, evaluation with token accounting, saved-result comparison and reports.
- `tests/`: tools, Agent, dataset and scoring tests.
- `docs/`: project documents and images.
- `requirements.txt`, `LICENSE`, `README.md`: pinned environment, license and usage documentation.

### Roadmap

- [ ] MCP wrapper: expose the existing tools to MCP-compatible clients.

The MCP wrapper is planned; structural screening and the project images are already implemented. Licensed under the [MIT License](LICENSE), Copyright (c) 2026 czhou143.

## 规则来源 / References

- **Brenk / BRENK** — Brenk et al., *Lessons Learnt from Assembling Screening Libraries for Drug Discovery for Neglected Diseases*. *ChemMedChem* (2008). DOI: [10.1002/cmdc.200700139](https://doi.org/10.1002/cmdc.200700139).
- **NIH** — Jadhav et al., *Quantitative analyses of aggregation, autofluorescence, and reactivity artifacts in a screen for inhibitors of a thiol protease*. *J Med Chem* (2010). DOI: [10.1021/jm901070c](https://doi.org/10.1021/jm901070c).
