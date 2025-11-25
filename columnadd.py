import json
import pandas as pd
from ase.io import read
from matminer.featurizers.structure.matrix import CoulombMatrix, SineCoulombMatrix
from pymatgen.io.ase import AseAtomsAdaptor

# -----------------------------
# 1️⃣ 读取 JSON
json_file = "AlonsoStrain2023_full_dataset.json"
with open(json_file) as f:
    data = json.load(f)

if 'reactions' not in data or 'edges' not in data['reactions']:
    raise ValueError("JSON does not contain 'reactions.edges' key!")

edges = data['reactions']['edges']
print(f"Total reactions: {len(edges)}")

# -----------------------------
# 2️⃣ 初始化库伦矩阵特征器（flatten=False）
cm_featurizer = CoulombMatrix(flatten=False)
scm_featurizer = SineCoulombMatrix(flatten=False)

# -----------------------------
# 3️⃣ 读取 extxyz 文件
extxyz_file = "AlonsoStrain2023_all.extxyz"
all_atoms = read(extxyz_file, index=':')  # 返回 ASE Atoms 对象列表

# 建立 system_name -> Atoms 对应字典
atoms_dict = {}
for atoms in all_atoms:
    sys_name = atoms.info.get('system_name', None)
    if sys_name is not None:
        atoms_dict[sys_name] = atoms

print(f"Total structures read from extxyz: {len(atoms_dict)}")

# -----------------------------
# 4️⃣ 遍历 edges，为每个 node 生成 Coulomb 矩阵
for i, entry in enumerate(edges):
    node = entry['node']
    node.setdefault('CoulombMatrix', [])
    node.setdefault('SineCoulombMatrix', [])

    if 'reactionSystems' not in node or len(node['reactionSystems']) == 0:
        print(f"Reaction {i} has no reactionSystems, skipping.")
        continue

    # 用 reactionSystems[0] 对应 extxyz
    system_name = node['reactionSystems'][0]['name']
    if system_name not in atoms_dict:
        print(f"Reaction {i}, system_name {system_name} not found in extxyz, skipping.")
        continue

    atoms = atoms_dict[system_name]

    # ASE -> pymatgen Structure
    structure = AseAtomsAdaptor.get_structure(atoms)

    # featurize 并压平
    cm_vec = cm_featurizer.featurize(structure)[0].ravel().tolist()
    scm_vec = scm_featurizer.featurize(structure)[0].ravel().tolist()
    node['CoulombMatrix'] = cm_vec
    node['SineCoulombMatrix'] = scm_vec

print("Coulomb matrices computed.")

# -----------------------------
# 5️⃣ 保存增强 JSON
enhanced_json_file = "AlonsoStrain2023_full_dataset_with_CM.json"
with open(enhanced_json_file, "w") as f:
    json.dump(data, f, indent=2)
print(f"Enhanced JSON saved to {enhanced_json_file}")

# -----------------------------
# 6️⃣ 保存 CSV
df = pd.DataFrame([entry['node'] for entry in edges])

# CoulombMatrix 拆成多列
if 'CoulombMatrix' in df.columns and df['CoulombMatrix'].apply(len).max() > 0:
    cm_cols = pd.DataFrame(df['CoulombMatrix'].to_list())
    cm_cols.columns = [f'CM_{i}' for i in range(cm_cols.shape[1])]
    df = pd.concat([df.drop(columns=['CoulombMatrix']), cm_cols], axis=1)

# SineCoulombMatrix 拆成多列
if 'SineCoulombMatrix' in df.columns and df['SineCoulombMatrix'].apply(len).max() > 0:
    scm_cols = pd.DataFrame(df['SineCoulombMatrix'].to_list())
    scm_cols.columns = [f'SCM_{i}' for i in range(scm_cols.shape[1])]
    df = pd.concat([df.drop(columns=['SineCoulombMatrix']), scm_cols], axis=1)

csv_file = "AlonsoStrain2023_full_dataset_with_CM.csv"
df.to_csv(csv_file, index=False)
print(f"Enhanced CSV saved to {csv_file}")
