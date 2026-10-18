# Chem Agent · 化学工具助手

用 RDKit 计算分子性质，用 DeepSeek 选择工具并生成中文回答；包含可复现的 40 题评测。
A small chemistry assistant with RDKit tools, a handwritten DeepSeek function-calling loop, and a reproducible 40-question evaluation.

支持 / Tools: `mol_weight`, `logp`, `formula`, `tpsa`, `hbd_hba`, `lipinski`, `similarity`.

## 运行 / Quick start

在项目根目录使用 Windows PowerShell；本机验证环境为 Python 3.12.5。
Run these commands from the project root in Windows PowerShell (tested with Python 3.12.5).

```powershell
# 首次安装 / First-time setup
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 每次新终端先激活；密钥只保存在环境变量中。
# Activate the environment in each new terminal; keep the key in the environment.
$env:LLM_API_KEY = "<your DeepSeek API key>"
python -m chem_agent.agent "阿司匹林的分子量是多少"
```

模型 / Model: `deepseek-chat`; API: `https://api.deepseek.com`.

## 测试与评测 / Tests and evaluation

```powershell
python -m pytest -q
python eval/build_questions.py --check
python eval/run_eval.py --dry-run --limit 3
python eval/run_eval.py --limit 5
python eval/run_eval.py
```

`--dry-run` 不需要密钥，不联网，用 mock 检查流程；报告单独写到 `eval/results/dry_run/`，不代表模型成绩。
Dry runs use expected answers as mock responses, require no API key, and save separate reports; their scores do not measure model quality.

真实评测逐题调用 Agent，异常记为失败并继续，输出带时间的 JSONL 详情和 `eval/results/summary.md`。
Live evaluation calls the Agent for each question, records individual exceptions as failures, and writes timestamped JSONL details plus a Markdown summary.

## 首次真实结果 / First live results

2026-10-18，完整 40 题，`deepseek-chat`，共 80 次模型请求；全部本地测试 **93 passed**。
Full live run on 2026-10-18: 40 questions, 80 model requests; **93 local tests passed**.

| 指标 / Metric | 结果 / Result |
| --- | ---: |
| 总准确率 / Overall accuracy | 100.00% (40/40) |
| 工具选择准确率 / Tool selection accuracy | 100.00% (40/40) |
| 数值或结果准确率 / Result accuracy | 100.00% (40/40) |
| 异常 / Exceptions | 0 |
| 错题 / Failed questions | 0 |

分子量、LogP、分子式、TPSA、氢键数、Lipinski 各 5 题，相似度 6 题，非法 SMILES 4 题；各类均全部通过。
Five questions each for molecular weight, LogP, formula, TPSA, hydrogen bonds, and Lipinski; six for similarity and four invalid SMILES. Every category passed.

有效题检查第一条工具调用的名称与结果：浮点数按绝对容差比较，整数、字符串、布尔值精确匹配。非法题检查中文回答是否明确说明输入无效，不限定所用工具；此项采用规则匹配。总通过还要求已有最终回答。该成绩只反映此题集与评分规则，不单独验证最终自然语言解释的每个数值或科学表述。
Valid questions score the first tool name and its result using absolute float tolerances and exact integer/string/boolean matching. Invalid-input questions use Chinese text rules to check explicit rejection, with no required tool. Overall success also requires a final answer. These scores apply to this dataset and rubric, not to every numerical or scientific claim in the final prose.

- [完整报告 / Full report](eval/results/summary.md)
- [真实逐题记录 / Live records](eval/results/20261018_221950_033160.jsonl)
- [当前进度 / Project progress](docs/03-当前进度.md)
