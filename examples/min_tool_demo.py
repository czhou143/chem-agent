"""最小 RDKit 工具示例：验证环境可用（不需要大模型）。运行：python examples/min_tool_demo.py"""
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


def mol_info(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "valid": False, "error": "无效的 SMILES"}
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd, hba = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {"smiles": smiles, "valid": True, "mol_weight": round(mw, 2), "logp": round(logp, 2),
            "hbd": hbd, "hba": hba, "lipinski_pass": violations <= 1}


if __name__ == "__main__":
    for s in ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "not_a_smiles"]:  # 阿司匹林、乙醇、错误输入
        print(mol_info(s))
