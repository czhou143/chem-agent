# Chem Agent | Chemistry tools and Agent evaluation

## Goal

A resume portfolio project for **ByteIntern, Agent development/evaluation**, connecting my chemistry background with practical Agent engineering. The emphasis is on runnable software, inspectable evidence, and explicit limits. [Project brief](01-需求文档.md)

## Architecture

A DeepSeek function-calling loop selects tools, receives local RDKit results, and produces an answer. The **8 tools** cover molecular weight, LogP, formula, TPSA, hydrogen-bond counts, Lipinski screening, similarity, and `safety_check`. [Tool registry](../chem_agent/tools.py)

HBA uses RDKit `NumHAcceptors` (caffeine: **3**); the original Lipinski N+O count gives **6** for caffeine's C8H10N4O2 formula. The implementation is unchanged. [Definitions and records](../README.md)

**One toolset, two interfaces:** the Agent calls functions directly through function calling; an official-SDK MCP stdio server exposes the same functions to external clients. Chemistry logic is reused. **Cursor integration tested** ([tool list](images/cursor-mcp-tools.webp) / [call screenshot](images/cursor-mcp-call.webp)); **Claude Desktop untested.** [Architecture](../README.md)

## Verified results

| Validation | Recorded result and evidence |
| --- | --- |
| v1 | **40/40**, `deepseek-chat`, **2026-10-18**. [Report](../eval/results/summary.md) |
| v2, before and after | **60/60 in each run**, `deepseek-flash`, non-thinking mode. [Before](../eval/results/v2_s4_20261019/before/summary.md) / [After](../eval/results/v2_s4_20261019/after/summary.md) |
| Offline tests | **198 passed**, covering tools, Agent, evaluation, and MCP. [Validation record](03-当前进度.md) |

The iteration changed one prompt sentence: q038/q039 each went from **2 tool calls to 1**. [Review](../eval/results/v2_error_analysis.md) Accuracy stayed unchanged; total tokens increased by **3,702** (**179,094 → 182,796**). [Comparison](../eval/results/v2_comparison.md) This was a reduction in redundant calls, not an accuracy gain. Different datasets and models prevent direct v1/v2 score comparisons.

## Safety boundary

`safety_check` uses Brenk and NIH catalogs: DOI **10.1002/cmdc.200700139** and **10.1021/jm901070c**. [Sources](../README.md)

Its disclaimer states that structural screening is not a toxicity or explosivity assessment and directs users to MSDS/SDS. System instructions refuse synthesis routes or preparation methods; MCP hosts must also enforce these instructions.

## My contribution

I confirmed requirements and acceptance criteria, used Codex to implement code and tests, coordinated live evaluations and result checks, and organized documentation and demonstration evidence. I collaborated with Haru on GitHub publication. Implementation was Codex-assisted. [Project record](03-当前进度.md) / [Raw demo log](../examples/demo_output.txt)

## Limitations

The questions remain relatively simple. Automated scoring does not verify every number in natural-language explanations. Structural alerts are not toxicity conclusions, and an empty alert list does not establish safety. [Scoring boundaries](../eval/results/summary.md)
