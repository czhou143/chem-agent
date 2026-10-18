# Codex 分步指令（共 10 条，按顺序一条条复制）

使用方法：在项目文件夹 chem-agent 里打开 Codex，**一次只贴一条**；做完后按“验收方法”自己检查，通过了再贴下一条。看不懂的地方直接问 Codex“请用化学专业新手能懂的话解释这段代码”。
已验证：2026-10-18 在 Linux + Python 3 上 `pip install rdkit`（2026.09.1）可直接安装，examples/min_tool_demo.py 运行正常。

---
## 第 1 条：搭建环境
```
我是化学专业学生，Python 忘了不少，没用过 Git。请在当前文件夹 chem-agent 里帮我搭建 Python 开发环境，每一步都先用中文告诉我你要做什么、为什么：
1. 检查我电脑上的 Python 版本（需要 3.10 或以上），没有的话告诉我怎么安装。
2. 创建虚拟环境 .venv，并告诉我在我的系统上怎么激活它。
3. 在虚拟环境里安装 rdkit、openai、pytest，并把它们写进 requirements.txt。
4. 创建 .gitignore，忽略 .venv/、__pycache__/、.env。
5. 运行 python examples/min_tool_demo.py，确认能输出阿司匹林的分子量约 180.16。
6. 最后用 3 句话总结：虚拟环境是什么、requirements.txt 有什么用、我以后每次开工要先执行哪条命令。
不要写其他功能代码。
```
**你需要理解的知识点**：虚拟环境（每个项目独立的一套 Python 包）；pip 安装包；requirements.txt；终端里“当前目录”的概念。
**验收方法**：新开一个终端，激活 .venv 后运行 `python -c "import rdkit; print(rdkit.__version__)"` 能打印版本号；`python examples/min_tool_demo.py` 输出阿司匹林 mol_weight 180.16。

---
## 第 2 条：第一次用 Git
```
请教我用 Git 管理这个项目，并且边做边解释：
1. 告诉我 git status、git add、git commit、git log 分别是干什么的，用“实验记录本”打比方。
2. 帮我设置 git 的用户名和邮箱（问我要，不要编）。
3. 把第 1 步新增的 requirements.txt 和 .gitignore 提交，提交信息写 "chore: 搭建开发环境"。
4. 运行 git log --oneline 给我看结果。
不要执行 git push，也不要修改历史提交。
```
**知识点**：仓库、暂存区、提交（commit）；为什么 .venv 不能提交；提交信息要写清楚。
**验收方法**：`git log --oneline` 能看到 "chore: 搭建开发环境"；`git status` 显示 working tree clean。

---
## 第 3 条：写工具层
```
请阅读 docs/01-需求文档.md 和 docs/02-软件架构文档.md，然后创建 chem_agent/__init__.py 和 chem_agent/tools.py，用 RDKit 实现 5 个工具函数：
- validate_smiles(smiles)：判断 SMILES 是否合法，返回规范化 SMILES
- mol_weight(smiles)：分子量，保留 2 位小数
- logp(smiles)：Crippen LogP，保留 2 位小数
- lipinski(smiles)：返回分子量、LogP、氢键供体数、受体数、违规条数、是否通过（违规≤1 视为通过）
- substructure_match(smiles, pattern_smarts)：判断分子是否含某子结构（比如苯环、羧基）
要求：每个函数都返回 dict；SMILES 无效时返回 {"error": "..."} 而不是抛异常；每个函数写中文 docstring；可以参考 examples/min_tool_demo.py。
写完后用阿司匹林 CC(=O)Oc1ccccc1C(=O)O 演示每个函数的输出，并逐行解释 mol_weight 函数。
```
**知识点**：SMILES 和 SMARTS 的区别；RDKit 的 Mol 对象；Lipinski 五规则（你的化学知识）；函数、参数、返回值、dict。
**验收方法**：阿司匹林分子量 180.16、含苯环为 True；输入 "abc" 返回 error 而程序不崩溃。

---
## 第 4 条：单元测试 + 工具描述
```
1. 在 tests/test_tools.py 用 pytest 为 tools.py 的 5 个函数各写至少 2 个测试（含一个非法 SMILES 的测试），用阿司匹林、乙醇、苯、咖啡因做例子，期望值请用 RDKit 实际算出来再写进测试。
2. 在 tools.py 末尾加一个 TOOLS 列表，按 OpenAI function calling 的 JSON schema 格式描述这 5 个工具（name、description、parameters），description 用英文写清楚何时该用这个工具；再加一个 TOOL_FUNCS 字典，把工具名映射到函数。
3. 运行 pytest -v，把结果给我看，并解释什么是 JSON schema、为什么大模型需要它。
4. 全部通过后 git commit，信息写 "feat: RDKit 工具层和单元测试"。
```
**知识点**：单元测试的意义；assert；function calling 的工具描述格式；description 写得好坏直接影响 LLM 选对工具。
**验收方法**：`pytest -v` 全绿（至少 10 个 passed）；git log 出现新提交。

---
## 第 5 条：Agent 主循环
```
请创建 chem_agent/agent.py，实现一个 function-calling Agent：
- 用 openai Python SDK，base_url 和模型名从环境变量 LLM_BASE_URL、LLM_MODEL 读取，API Key 从 LLM_API_KEY 读取（兼容 DeepSeek 等 OpenAI 兼容接口），绝对不要把 key 写进代码。
- 函数 run_agent(question, max_steps=5) 的流程：把问题和 TOOLS 发给模型 → 如果模型要求调用工具，就用 TOOL_FUNCS 执行并把结果作为 tool 消息发回 → 循环，直到模型给出最终答案或超过 max_steps。
- 返回 {"answer": 最终回答, "tool_calls": [按顺序调用过的工具名和参数]}。
- 加一个 if __name__ == "__main__" 的命令行入口：python -m chem_agent.agent "阿司匹林的分子量是多少？SMILES 是 CC(=O)Oc1ccccc1C(=O)O"
- 新建 .env.example 列出三个环境变量（只写占位符）。
写完后画一个文字流程图解释主循环，并说明 max_steps 为什么必要。
```
**知识点**：Agent = 大模型 + 工具 + 循环；messages 列表里 system/user/assistant/tool 四种角色；环境变量与密钥安全。
**验收方法**：设置好环境变量后运行命令行示例，回答中含 180.16，tool_calls 里出现 mol_weight；`git grep -n "sk-"` 没有任何结果。

---
## 第 6 条：准备评测集
```
请创建 eval/questions.jsonl，写 40 道化学题，每行一个 JSON：{"id", "question", "answer", "answer_type", "expected_tool", "category"}。
- answer_type 只能是 "number"（数值，允许 1% 误差）或 "bool" 或 "text"（精确匹配规范 SMILES）
- 覆盖：分子量 10 题、LogP 8 题、Lipinski 判断 8 题、子结构判断 8 题、SMILES 合法性 6 题；用常见药物和有机分子，题目里直接给出 SMILES
- 所有标准答案必须写一个脚本 eval/make_answers.py 用 tools.py 实际计算生成，不能凭记忆写
完成后随机抽 5 道题给我看，让我用化学知识检查题目是否合理。
```
**知识点**：评测集为什么要有标准答案；数值容差；为什么答案要程序生成（避免编造）；JSONL 格式。
**验收方法**：`wc -l eval/questions.jsonl` 为 40；你亲自检查 5 道题合理；重新跑 make_answers.py 结果不变。

---
## 第 7 条：评分脚本
```
请创建 eval/run_eval.py：
1. 逐题调用 run_agent，用正则或让模型按 "FINAL: <答案>" 格式输出，提取最终答案。
2. 按 answer_type 打分：number 允许 1% 相对误差，bool 比较真假，text 用 RDKit 规范化后比较。
3. 统计两个指标：答案准确率、工具选择正确率（第一个调用的工具是否等于 expected_tool），并按 category 分组输出。
4. 结果保存到 eval/results/日期_模型名.json，终端打印一张汇总表。
5. 支持 --limit 5 参数，方便先跑 5 题试试。
写完先用 --limit 5 运行，再解释准确率和工具正确率分别说明了什么问题。
```
**知识点**：自动化评测；为什么要两个指标；失败案例分析（面试必问“错在哪、怎么改进”）。
**验收方法**：`python eval/run_eval.py --limit 5` 正常输出表格；完整跑 40 题后得到真实准确率，results 下有 json 文件。

---
## 第 8 条：根据失败案例改进一次
```
请读取 eval/results 下最新的结果，列出所有答错的题，按原因分类（选错工具 / 参数错误 / 答案提取失败 / 模型自己算没调工具 等）。
针对最主要的一类问题提出 1 个改进（比如改 system prompt 或改工具 description），实施后重新跑完整评测，用表格对比改进前后的两个指标。
把对比结果写进 docs/03-当前进度.md，然后 git commit "eval: 第一次迭代"。
```
**知识点**：迭代优化；prompt 与工具描述对效果的影响；前后对比才有说服力。
**验收方法**：有前后两份结果文件和对比表；数字真实来自脚本输出。

---
## 第 9 条：写 README
```
请为项目写 README.md（中文为主，标题和小节附英文），包含：
1. 一句话介绍 + 架构图（用 mermaid）
2. 快速开始：clone、建虚拟环境、pip install -r requirements.txt、配置 .env、运行 demo、运行评测，每步一条命令
3. 工具列表表格
4. 评测结果：直接引用 eval/results 里最新的真实数字和改进前后对比，不要编造任何数字
5. 失败案例分析和后续计划（MCP 封装、安全检查工具）
6. 参考项目：ChemCrow、scientific-agent-skills、deer-flow（见 docs/04-调研.md）
写完后，请按 README 的步骤在一个新的临时目录里从头跑一遍，确认没有遗漏。
```
**知识点**：README 就是作品的“简历”；可复现性；为什么面试官会先看评测结果。
**验收方法**：在新目录按 README 操作能跑通 demo；README 里的数字和 results 文件一致。

---
## 第 10 条：推送到 GitHub
```
我已经注册了 GitHub 账号（用户名：<填你的用户名>），并在网页上新建了一个空的公开仓库 chem-agent（不勾选 README）。请一步步教我：
1. 检查 .env 和 .venv 没有被 git 跟踪，确认仓库里没有任何 API Key。
2. 配置 GitHub 登录方式（推荐 GitHub CLI 的 gh auth login 或 HTTPS + Personal Access Token），需要我在浏览器操作时停下来告诉我。
3. git remote add origin，然后 git push -u origin main。
4. 解释 remote、push、pull 是什么，以后修改代码后上传需要哪三条命令。
不要使用 --force。
```
**知识点**：本地仓库与远程仓库；身份验证；密钥泄露的后果（公开仓库里的 key 几分钟内就会被盗用）。
**验收方法**：浏览器打开 https://github.com/<你的用户名>/chem-agent 能看到代码和 README；仓库里搜不到 key。
