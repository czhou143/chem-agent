"""可复用的本地 RDKit 工具，不需要 API Key 或网络。

成功时返回含结果字段的字典，非法输入统一返回 {"error": "原因"}。
例如：mol_weight("CCO") 返回 {"mol_weight": 46.07}。
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors


def _parse_smiles(smiles: str) -> Chem.Mol | None:
    """把 SMILES 转成分子；下划线表示这是供本模块使用的辅助函数。"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    # 禁止把尾部的错误文字当成分子名称而悄悄接受。
    parameters = Chem.SmilesParserParams()
    parameters.parseName = False
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), parameters)
    except (TypeError, ValueError, RuntimeError):
        return None

    # RDKit 接受空分子，但这些性质工具需要至少一个原子。
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def _invalid_smiles(argument: str = "smiles") -> dict:
    return {"error": f"{argument} 不是有效的非空 SMILES 字符串"}


def mol_weight(smiles: str) -> dict:
    """返回平均分子量 mol_weight（数值对应摩尔质量 g/mol），保留两位小数。"""
    mol = _parse_smiles(smiles)
    if mol is None:
        return _invalid_smiles()
    return {"mol_weight": round(Descriptors.MolWt(mol), 2)}


def logp(smiles: str) -> dict:
    """返回 RDKit 的 Wildman-Crippen 预测 LogP，无单位。"""
    mol = _parse_smiles(smiles)
    if mol is None:
        return _invalid_smiles()
    return {"logp": Crippen.MolLogP(mol)}


def formula(smiles: str) -> dict:
    """返回分子式，例如阿司匹林的 C9H8O4。"""
    mol = _parse_smiles(smiles)
    if mol is None:
        return _invalid_smiles()
    return {"formula": rdMolDescriptors.CalcMolFormula(mol)}


def tpsa(smiles: str) -> dict:
    """返回拓扑极性表面积 tpsa，单位为平方埃（Å²）。"""
    mol = _parse_smiles(smiles)
    if mol is None:
        return _invalid_smiles()
    return {"tpsa": rdMolDescriptors.CalcTPSA(mol)}


def hbd_hba(smiles: str) -> dict:
    """返回氢键供体数 hbd 和氢键受体数 hba。"""
    mol = _parse_smiles(smiles)
    if mol is None:
        return _invalid_smiles()
    return {"hbd": Lipinski.NumHDonors(mol), "hba": Lipinski.NumHAcceptors(mol)}


def lipinski(smiles: str) -> dict:
    """返回 Lipinski 指标、违反项数和筛选结果；不代表药物安全性或疗效。

    阈值为分子量 <= 500、LogP <= 5、HBD <= 5、HBA <= 10。
    沿用 min_tool_demo.py：最多违反一项时 lipinski_pass 为 True。
    判断使用未经四舍五入的数值，避免在阈值附近误判。
    """
    mol = _parse_smiles(smiles)
    if mol is None:
        return _invalid_smiles()

    weight = Descriptors.MolWt(mol)
    logp_value = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([weight > 500, logp_value > 5, hbd > 5, hba > 10])
    return {
        "mol_weight": round(weight, 2),
        "logp": logp_value,
        "hbd": hbd,
        "hba": hba,
        "violations": violations,
        "lipinski_pass": violations <= 1,
    }


def similarity(smiles_a: str, smiles_b: str) -> dict:
    """返回 Morgan 指纹的 Tanimoto 相似度，范围 0 到 1。

    使用半径 2、2048 位的指纹，不编码手性；1 表示指纹相同，
    不能据此断言两个输入在所有化学性质上完全一致。
    """
    mol_a = _parse_smiles(smiles_a)
    if mol_a is None:
        return _invalid_smiles("smiles_a")
    mol_b = _parse_smiles(smiles_b)
    if mol_b is None:
        return _invalid_smiles("smiles_b")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    fingerprint_a = generator.GetFingerprint(mol_a)
    fingerprint_b = generator.GetFingerprint(mol_b)
    return {"similarity": DataStructs.TanimotoSimilarity(fingerprint_a, fingerprint_b)}


# 用字符串名称找到对应函数，例如 TOOLS["formula"]("CCO")。
TOOLS = {
    "mol_weight": mol_weight,
    "logp": logp,
    "formula": formula,
    "tpsa": tpsa,
    "hbd_hba": hbd_hba,
    "lipinski": lipinski,
    "similarity": similarity,
}


def _tool_schema(
    name: str, description: str, arguments: tuple[str, ...] = ("smiles",)
) -> dict:
    """生成 OpenAI Chat Completions tools 参数中的一个函数描述。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description + "；非法 SMILES 返回 error 字段。",
            "parameters": {
                "type": "object",
                "properties": {
                    argument: {
                        "type": "string",
                        "description": f"{argument}：非空 SMILES 字符串，例如 CCO",
                    }
                    for argument in arguments
                },
                "required": list(arguments),
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


# 这里只声明函数及其输入格式，不创建客户端、不调用大模型。
TOOL_SCHEMAS = [
    _tool_schema("mol_weight", "计算平均分子量，返回 mol_weight，保留两位小数"),
    _tool_schema("logp", "计算 Wildman-Crippen 预测 LogP，返回 logp"),
    _tool_schema("formula", "计算分子式，返回 formula"),
    _tool_schema("tpsa", "计算拓扑极性表面积（Å²），返回 tpsa"),
    _tool_schema("hbd_hba", "计算氢键供体和受体数，返回 hbd 和 hba"),
    _tool_schema(
        "lipinski",
        "计算 Lipinski 指标 mol_weight、logp、hbd、hba，返回 violations 和 "
        "lipinski_pass；超过 500、5、5、10 的指标各算一项违反，最多违反一项算通过",
    ),
    _tool_schema(
        "similarity",
        "计算两个分子的 Morgan 指纹 Tanimoto 相似度，返回 similarity（0 到 1）；"
        "指纹半径 2、2048 位、不编码手性",
        ("smiles_a", "smiles_b"),
    ),
]
