"""用本地 RDKit 工具生成、校验 40 道中文评测题，不调用模型或网络。

生成：python eval/build_questions.py
只校验已有文件：python eval/build_questions.py --check

expected_value 保存工具实际返回的完整字典，不手填答案。tolerance 为浮点数
评分时的绝对误差；整数、布尔值、分子式应精确匹配。生成器自身会精确重算，
以发现手改的答案。非法题的 expected_tool=null 表示不指定工具，并非禁止调用
工具；未来评分应检查回答是否说明输入无效，而不是逐字匹配 error 的措辞。
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

# 直接运行脚本时，让 Python 能找到同级目录中的 chem_agent 包。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import rdBase

from chem_agent import tools


OUTPUT_PATH = Path(__file__).resolve().with_name("questions.jsonl")
FIELDS = {"id", "question", "expected_tool", "expected_value", "tolerance"}

# 名称/结构参考 PubChem：药物使用中性分子，布洛芬不指定对映异构体。
# https://pubchem.ncbi.nlm.nih.gov/compound/Aspirin (CID 2244)
# https://pubchem.ncbi.nlm.nih.gov/compound/Ibuprofen (CID 3672)
# https://pubchem.ncbi.nlm.nih.gov/compound/Caffeine (CID 2519)
# https://pubchem.ncbi.nlm.nih.gov/compound/Acetaminophen (CID 1983)
# https://pubchem.ncbi.nlm.nih.gov/compound/dextrose (CID 5793)
# https://pubchem.ncbi.nlm.nih.gov/compound/cholesterol (CID 5997)
# https://pubchem.ncbi.nlm.nih.gov/compound/Atorvastatin (CID 60823)
# 这里只定义输入；所有标准答案均在 build_questions() 中通过 TOOLS 计算。
MOLECULES = {
    "aspirin": ("阿司匹林", "CC(=O)Oc1ccccc1C(=O)O"),
    "ibuprofen": ("布洛芬", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    "caffeine": ("咖啡因", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
    "paracetamol": ("对乙酰氨基酚", "CC(=O)Nc1ccc(O)cc1"),
    "ethanol": ("乙醇", "CCO"),
    "ethanol_alt": ("乙醇的另一种 SMILES 写法", "OCC"),
    "benzene": ("苯", "c1ccccc1"),
    "phenol": ("苯酚", "Oc1ccccc1"),
    "glucose": (
        "D-吡喃葡萄糖（异头构型未指定）",
        "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
    ),
    "cholesterol": (
        "胆固醇",
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4"
        "[C@@]3(CC[C@@H](C4)O)C)C",
    ),
    "atorvastatin": (
        "阿托伐他汀（中性游离酸）",
        "CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)"
        "C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    ),
}

SINGLE_TOOL_CASES = {
    "mol_weight": ("aspirin", "ibuprofen", "caffeine", "paracetamol", "ethanol"),
    "logp": ("aspirin", "ibuprofen", "caffeine", "ethanol", "benzene"),
    "formula": ("aspirin", "caffeine", "paracetamol", "ethanol", "glucose"),
    "tpsa": ("aspirin", "ibuprofen", "caffeine", "paracetamol", "glucose"),
    "hbd_hba": ("aspirin", "caffeine", "paracetamol", "benzene", "glucose"),
    # 包括零项、一项和多项违反的情况，避免所有筛选题都通过。
    "lipinski": ("aspirin", "ibuprofen", "caffeine", "cholesterol", "atorvastatin"),
}
PROMPTS = {
    "mol_weight": "{name}的分子量是多少？请保留两位小数。SMILES：`{smiles}`。",
    "logp": "请计算{name}的 Wildman-Crippen 预测 LogP。SMILES：`{smiles}`。",
    "formula": "请给出{name}的分子式。SMILES：`{smiles}`。",
    "tpsa": "{name}的拓扑极性表面积 TPSA 是多少平方埃？SMILES：`{smiles}`。",
    "hbd_hba": "{name}的氢键供体数 HBD 和受体数 HBA 各是多少？SMILES：`{smiles}`。",
    "lipinski": "请判断{name}是否通过本项目的 Lipinski 筛选（最多违反一项算通过），"
    "列出分子量、LogP、HBD、HBA 及违反项数。SMILES：`{smiles}`。",
}
SIMILARITY_PAIRS = (
    ("aspirin", "aspirin"),
    ("ethanol", "ethanol_alt"),
    ("benzene", "phenol"),
    ("caffeine", "aspirin"),
    ("ibuprofen", "paracetamol"),
    ("ethanol", "benzene"),
)
INVALID_SMILES = ("not_a_smiles", "C1CC", "C(C)(C)(C)(C)C", "")
TOLERANCES = {
    "mol_weight": 0.01,
    "logp": 0.01,
    "formula": 0,
    "tpsa": 0.01,
    "hbd_hba": 0,
    "lipinski": 0.01,
    "similarity": 0.000001,
    None: 0,
}


def _question_specs() -> list[dict]:
    """把输入清单展开成题干和工具参数，题干中的 SMILES 与计算输入同源。"""
    specs = []
    for tool_name, molecule_names in SINGLE_TOOL_CASES.items():
        for molecule in molecule_names:
            name, smiles = MOLECULES[molecule]
            specs.append({
                "question": PROMPTS[tool_name].format(name=name, smiles=smiles),
                "tool": tool_name,
                "arguments": {"smiles": smiles},
            })

    for molecule_a, molecule_b in SIMILARITY_PAIRS:
        name_a, smiles_a = MOLECULES[molecule_a]
        name_b, smiles_b = MOLECULES[molecule_b]
        specs.append({
            "question": f"请计算{name_a}（SMILES A：`{smiles_a}`）与{name_b}"
            f"（SMILES B：`{smiles_b}`）的 Morgan 指纹 Tanimoto 相似度，"
            "使用半径 2、2048 位且不编码手性的指纹。",
            "tool": "similarity",
            "arguments": {"smiles_a": smiles_a, "smiles_b": smiles_b},
        })

    for smiles in INVALID_SMILES:
        displayed = json.dumps(smiles, ensure_ascii=False)
        specs.append({
            "question": f"请检查 SMILES 字符串 {displayed} 是否有效；"
            "如果输入无效，请明确说明，不要猜测或擅自补全分子结构。",
            "tool": None,
            "arguments": {"smiles": smiles},
        })
    return specs


def build_questions() -> list[dict]:
    """调用现有工具生成标准答案；每次调用都重新计算，不读取旧题库。"""
    rows = []
    for index, spec in enumerate(_question_specs(), start=1):
        tool_name = spec["tool"]
        if tool_name is None:
            # 用现有解析路径验证非法输入；只在这些预期失败的调用中屏蔽日志。
            with rdBase.BlockLogs():
                result = tools.TOOLS["formula"](**spec["arguments"])
            if not isinstance(result.get("error"), str) or not result["error"]:
                raise ValueError(f"非法输入案例意外被工具接受：{spec['arguments']}")
        else:
            result = tools.TOOLS[tool_name](**spec["arguments"])
            if "error" in result:
                raise ValueError(f"有效题目的计算失败：{tool_name}，{result['error']}")

        rows.append({
            "id": f"q{index:03d}",
            "question": spec["question"],
            "expected_tool": tool_name,
            "expected_value": result,
            "tolerance": TOLERANCES[tool_name],
        })
    return rows


def validate_questions(rows: list[dict]) -> None:
    """检查结构、覆盖范围和来源；重算答案并精确对照，发现篡改或版本漂移。"""
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("题库必须恰好包含 40 道题。")

    ids = set()
    questions = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("每道题必须且只能包含 id/question/expected_tool/expected_value/tolerance。")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
            raise ValueError("题目 id 必须为唯一的非空字符串。")
        ids.add(row["id"])
        question = row["question"]
        if (not isinstance(question, str) or not question.strip()
                or not any("\u4e00" <= char <= "\u9fff" for char in question)
                or question in questions):
            raise ValueError(f"{row['id']} 必须有唯一的中文题干。")
        questions.add(question)

        tool_name = row["expected_tool"]
        if tool_name is not None and (not isinstance(tool_name, str) or tool_name not in tools.TOOLS):
            raise ValueError(f"{row['id']} 的 expected_tool 不在 TOOLS 中。")
        value = row["expected_value"]
        if not isinstance(value, dict) or not value:
            raise ValueError(f"{row['id']} 的 expected_value 必须是工具返回的非空字典。")
        if tool_name is None:
            if not isinstance(value.get("error"), str) or not value["error"].strip():
                raise ValueError(f"{row['id']} 的非法输入必须有 error 结果。")
        elif "error" in value:
            raise ValueError(f"{row['id']} 的有效输入不应返回 error。")
        tolerance = row["tolerance"]
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError(f"{row['id']} 的 tolerance 必须是非负有限数。")

    counts = Counter(row["expected_tool"] for row in rows)
    expected_counts = {name: 5 for name in tools.TOOLS}
    expected_counts.update({"similarity": 6, None: 4})
    if counts != expected_counts:
        raise ValueError("题目分布应为六类各 5 题、相似度 6 题、非法输入 4 题。")

    # JSON 精确比较也能区分 true 与 1、1 与 1.0，且禁止 NaN/Infinity。
    for row, reference in zip(rows, build_questions()):
        try:
            actual = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
            expected = json.dumps(reference, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{row['id']} 含有不合法的 JSON 值。") from exc
        if actual != expected:
            raise ValueError(f"{row['id']} 与题目模板或当前工具计算结果不一致，请重新生成。")


def _read_questions() -> list[dict]:
    lines = OUTPUT_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def main() -> int:
    parser = argparse.ArgumentParser(description="用 RDKit 生成并校验 40 道中文评测题")
    parser.add_argument("--check", action="store_true", help="只校验已有 questions.jsonl，不改写文件")
    args = parser.parse_args()
    try:
        if args.check:
            rows = _read_questions()
            validate_questions(rows)
        else:
            rows = build_questions()
            validate_questions(rows)
            text = "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in rows) + "\n"
            OUTPUT_PATH.write_text(text, encoding="utf-8", newline="\n")
            validate_questions(_read_questions())
    except (OSError, ValueError) as exc:
        print(f"题库生成或校验失败：{exc}", file=sys.stderr)
        return 1

    action = "校验通过" if args.check else "已生成并校验"
    print(f"{action}：{OUTPUT_PATH}（{len(rows)} 题）")
    for tool_name, count in Counter(row["expected_tool"] for row in rows).items():
        print(f"  {tool_name or '非法 SMILES'}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
