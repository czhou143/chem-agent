"""保留 v1 的 40 题，生成并校验 v2 的 60 题，不调用模型或网络。

生成 v2（不改写 v1）：python eval/build_questions.py
同时校验 v1/v2：python eval/build_questions.py --check

expected_value 来自工具实算：单工具保存字典，安全题提取警示名称列表，
多工具按调用顺序保存结果字典列表，不手填答案。tolerance 为浮点数
评分时的绝对误差；整数、布尔值、分子式应精确匹配。生成器自身会精确重算，
以发现手改的答案。非法题的 expected_tool=null 表示不指定工具，并非禁止调用
工具；未来评分应检查回答是否说明输入无效，而不是逐字匹配 error 的措辞。
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
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
V2_OUTPUT_PATH = Path(__file__).resolve().with_name("questions_v2.jsonl")
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

# v2 只定义输入和提问方式，不存放手填的标准答案。
SAFETY_CASES = (
    ("2,4,6-三硝基甲苯（TNT）", "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"),
    MOLECULES["ethanol"],
    MOLECULES["glucose"],
    ("乙酰氯", "CC(=O)Cl"),
    ("叠氮甲烷", "CN=[N+]=[N-]"),
    ("硝基苯", "O=[N+]([O-])c1ccccc1"),
    ("异氰酸甲酯", "CN=C=O"),
    ("环氧乙烷", "C1CO1"),
)
MULTI_STEP_CASES = (
    (("mol_weight", "aspirin"), ("lipinski", "aspirin"),
     "请先调用分子量工具，再调用 Lipinski 工具，列出两次结果并判断筛选是否通过。"),
    (("logp", "ibuprofen"), ("logp", "caffeine"),
     "请先计算布洛芬的 LogP，再计算咖啡因的 LogP，列出两次结果并比较谁更大。"),
    (("formula", "ethanol"), ("mol_weight", "ethanol"),
     "请先查询分子式，再计算分子量，分别标注结果，避免把两种信息混在一起。"),
    (("tpsa", "glucose"), ("hbd_hba", "glucose"),
     "请先计算 TPSA，再统计氢键供体和受体数，依次列出结果。"),
    (("hbd_hba", "caffeine"), ("tpsa", "caffeine"),
     "请先统计氢键供体和受体数，再计算 TPSA，依次列出结果。"),
    (("formula", "paracetamol"), ("lipinski", "paracetamol"),
     "请先查询分子式，再调用 Lipinski 工具完成筛选，依次列出结果。"),
)
CONFUSABLE_CASES = (
    ("formula", "aspirin", "我需要元素组成的分子式，不是以 g/mol 表示的分子量，请只查询分子式。"),
    ("mol_weight", "caffeine", "我需要以 g/mol 表示的分子量，不是 C、H、N、O 个数组成的分子式。"),
    ("tpsa", "ibuprofen", "请给出单位为平方埃的拓扑极性表面积 TPSA，不要给脂水分配系数 LogP。"),
    ("logp", "ethanol", "我要 Wildman-Crippen 预测 LogP，不是以平方埃计的 TPSA。"),
    ("hbd_hba", "aspirin", "重点告诉我氢键供体 HBD 有几个，同时标注受体 HBA，注意不要把供体和受体的数目交换。"),
    ("hbd_hba", "caffeine", "重点告诉我氢键受体 HBA 有几个，同时标注供体 HBD，不要将受体回答成供体。"),
)


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
    # v1 覆盖由题目模板定义，工具注册表后续扩展不会改变已有 40 题。
    expected_counts = {name: 5 for name in SINGLE_TOOL_CASES}
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


def _read_questions(path=None) -> list[dict]:
    lines = (OUTPUT_PATH if path is None else Path(path)).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _calculate(tool_name: str, smiles: str) -> dict:
    result = tools.TOOLS[tool_name](smiles)
    if "error" in result:
        raise ValueError(f"v2 输入计算失败：{tool_name}，{result['error']}")
    return result


def build_questions_v2() -> list[dict]:
    """复制并重算校验 v1，追加 8 道安全、6 道多工具、6 道易混淆题。"""
    base = _read_questions()
    validate_questions(base)
    rows = deepcopy(base)

    def append(question, tool_name, value, category, tolerance, expected_tools=None):
        row = {"id": f"q{len(rows) + 1:03d}", "question": question,
               "expected_tool": tool_name, "expected_value": value,
               "tolerance": tolerance, "category": category}
        if expected_tools is not None:
            row["expected_tools"] = expected_tools
        rows.append(row)

    for name, smiles in SAFETY_CASES:
        result = _calculate("safety_check", smiles)
        names = sorted(alert["name"] for alert in result["alerts"])
        append(f"请对{name}做安全相关的药物化学结构警示筛查，列出命中的警示名称；"
               f"没有警示时明确说明，并转述筛查免责声明。SMILES：`{smiles}`。",
               "safety_check", names, "safety", 0)

    for first, second, prompt in MULTI_STEP_CASES:
        steps = (first, second)
        inputs = "；".join(f"第{index}步：{MOLECULES[molecule][0]}，SMILES：`{MOLECULES[molecule][1]}`"
                          for index, (_, molecule) in enumerate(steps, 1))
        results = [_calculate(name, MOLECULES[molecule][1]) for name, molecule in steps]
        append(f"{prompt}请按要求顺序完成两次工具调用。{inputs}。", first[0], results,
               "multi_step", max(TOLERANCES[name] for name, _ in steps),
               expected_tools=[name for name, _ in steps])

    for tool_name, molecule, prompt in CONFUSABLE_CASES:
        name, smiles = MOLECULES[molecule]
        append(f"{name}，SMILES：`{smiles}`。{prompt}", tool_name,
               _calculate(tool_name, smiles), "confusable", TOLERANCES[tool_name])
    return rows


def validate_questions_v2(rows: list[dict]) -> None:
    """对照模板和工具重算结果校验全部字段；v2 的前 40 题保持 v1 格式。"""
    if not isinstance(rows, list) or len(rows) != 60:
        raise ValueError("v2 必须恰好包含 60 道题。")
    if not all(isinstance(row, dict) and isinstance(row.get("id"), str)
               and isinstance(row.get("question"), str) for row in rows):
        raise ValueError("v2 每道题必须为字典，id 和 question 必须为字符串。")
    validate_questions(rows[:40])
    if len({row.get("id") for row in rows}) != 60:
        raise ValueError("v2 题目 id 必须唯一。")
    if len({row.get("question") for row in rows}) != 60:
        raise ValueError("v2 题干必须唯一。")
    reference = build_questions_v2()
    for row, expected in zip(rows, reference):
        if json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) != json.dumps(
            expected, sort_keys=True, ensure_ascii=False, allow_nan=False
        ):
            raise ValueError(f"v2 的 {expected['id']} 与模板或工具实算结果不一致。")


def main() -> int:
    parser = argparse.ArgumentParser(description="保留 v1，生成 60 题 v2；输出 v2 文件 SHA256")
    parser.add_argument("--check", action="store_true", help="同时校验已有 v1 和 v2，不改写文件")
    args = parser.parse_args()
    try:
        rows = _read_questions()
        validate_questions(rows)
        if not args.check:
            v2_rows = build_questions_v2()
            validate_questions_v2(v2_rows)
            text = "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in v2_rows) + "\n"
            V2_OUTPUT_PATH.write_text(text, encoding="utf-8", newline="\n")
        validate_questions_v2(_read_questions(V2_OUTPUT_PATH))
    except (OSError, ValueError) as exc:
        print(f"题库生成或校验失败：{exc}", file=sys.stderr)
        return 1

    print(f"v1 校验通过：{OUTPUT_PATH}（40 题，未改写）")
    action = "校验通过" if args.check else "已生成并校验"
    print(f"v2 {action}：{V2_OUTPUT_PATH}（60 题）")
    print("v2 分布：原 40 题 + 安全检查 8 题 + 多步骤 6 题 + 易混淆 6 题")
    print(f"v2 SHA256：{hashlib.sha256(V2_OUTPUT_PATH.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
