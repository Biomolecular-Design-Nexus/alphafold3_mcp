"""Patch features.py to use srETKDGv3 instead of ETKDGv3 for conformer generation.

ETKDGv3 hangs indefinitely on large macrocyclic peptides.
srETKDGv3 (small ring variant) handles macrocycles correctly.
"""

FEATURES_PY = "/app/repo/alphafold3/src/alphafold3/model/features.py"

OLD = """\
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_state.randint(1, 1 << 31))
    AllChem.EmbedMolecule(m, params)"""

NEW = """\
    seed = int(random_state.randint(1, 1 << 31))
    params = AllChem.srETKDGv3()
    params.randomSeed = seed
    embed_result = AllChem.EmbedMolecule(m, params)
    if embed_result == -1:
      # Fallback to ETKDGv3 for non-macrocyclic molecules
      params = AllChem.ETKDGv3()
      params.randomSeed = seed
      AllChem.EmbedMolecule(m, params)"""

text = open(FEATURES_PY).read()
assert OLD in text, "Pattern not found in features.py - maybe already patched"
text = text.replace(OLD, NEW)
open(FEATURES_PY, "w").write(text)
print("srETKDGv3 patch applied successfully")
