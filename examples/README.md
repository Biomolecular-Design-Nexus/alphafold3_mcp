# AlphaFold3 MCP Examples

This directory contains example scripts demonstrating common AlphaFold3 protein structure prediction use cases using the MCP server.

## Use Case Scripts

| Script | Description | Example Command |
|--------|-------------|-----------------|
| `use_case_1_simple_protein.py` | Simple protein structure prediction | `python examples/use_case_1_simple_protein.py --fasta examples/data/sample_protein.fasta --name "kinase"` |
| `use_case_2_protein_ligand.py` | Protein-ligand complex prediction | `python examples/use_case_2_protein_ligand.py --fasta examples/data/sample_protein.fasta --smiles "CCO" --name "kinase_ethanol"` |
| `use_case_3_precomputed_msa.py` | Fast prediction with pre-computed MSA | `python examples/use_case_3_precomputed_msa.py --fasta examples/data/wt.fasta --msa examples/data/wt.a3m --name "subtilisin_fast"` |
| `use_case_4_batch_variants.py` | Batch prediction of protein variants | `python examples/use_case_4_batch_variants.py --fasta examples/data/protein_variants.fasta --output variants_results` |

## Demo Data

| File | Description | Size |
|------|-------------|------|
| `data/sample_protein.fasta` | Sample protein kinase sequence | 1 sequence |
| `data/protein_variants.fasta` | Multiple kinase variants | 3 sequences |
| `data/sample_mutations.txt` | Example mutation list | 5 mutations |
| `data/wt.fasta` | Subtilisin wild-type sequence | 1 sequence |
| `data/subtilisin_variants.fasta` | Subtilisin variants | 116 sequences |
| `data/wt.a3m` | Pre-computed MSA for subtilisin | ~2.8MB |

## Quick Start Examples

### 1. Simple Protein Prediction
```bash
# Predict structure from sequence
python examples/use_case_1_simple_protein.py \
    --fasta examples/data/sample_protein.fasta \
    --name "sample_kinase" \
    --output results_simple

# This creates: results_simple/input.json
```

### 2. Protein-Ligand Complex
```bash
# Predict protein bound to ethanol
python examples/use_case_2_protein_ligand.py \
    --fasta examples/data/sample_protein.fasta \
    --smiles "CCO" \
    --name "kinase_ethanol" \
    --output results_complex

# Using a more complex ligand (from 1IEP example)
python examples/use_case_2_protein_ligand.py \
    --fasta examples/data/sample_protein.fasta \
    --smiles "Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1" \
    --name "kinase_inhibitor" \
    --output results_complex_inhibitor
```

### 3. Fast Prediction with MSA
```bash
# Use pre-computed MSA for faster prediction
python examples/use_case_3_precomputed_msa.py \
    --fasta examples/data/wt.fasta \
    --msa examples/data/wt.a3m \
    --name "subtilisin_fast" \
    --mode a3m \
    --output results_msa
```

### 4. Batch Variants
```bash
# Predict multiple variants from FASTA
python examples/use_case_4_batch_variants.py \
    --fasta examples/data/protein_variants.fasta \
    --output results_batch

# Generate variants from mutation list
python examples/use_case_4_batch_variants.py \
    --variants examples/data/sample_mutations.txt \
    --template examples/data/sample_protein.fasta \
    --output results_mutations
```

## Running Predictions

All scripts create JSON configuration files but do not run predictions by default. To run AlphaFold3:

1. **Activate environment:**
   ```bash
   mamba activate ./env
   ```

2. **Run prediction using MCP tools:**
   ```bash
   python src/alphafold3_mcp.py
   # Then use the MCP server tools
   ```

3. **Or run directly:**
   ```bash
   python src/tools/af3_predict_structure.py \
       --json_path=results_simple/input.json \
       --model_dir=repo/alphafold3/model \
       --db_dir=repo/alphafold3/alphafold3_db \
       --output_dir=results_simple
   ```

## Prediction Modes

- **default**: Full MSA search + template search + inference (most accurate, slowest)
- **a3m**: Pre-computed A3M MSA + optional template search + inference (moderate speed)
- **msa**: Pre-computed MSA/templates, inference only (fastest, good for variants)

## Output Files

Each prediction generates:
- `{name}_model.cif` - 3D structure in mmCIF format
- `{name}_confidences.json` - Per-residue confidence scores
- `{name}_data.json` - Full prediction data
- `{name}_summary_confidences.json` - Summary confidence metrics

## Tips

1. **For protein engineering**: Use mode="msa" with pre-computed MSA for variant analysis
2. **For novel proteins**: Use default mode for best accuracy
3. **For complexes**: Allow extra time as ligand sampling increases runtime
4. **For large batches**: Consider parallel execution with multiple GPUs

## Real Examples

The `1iep/`, `subtilisin/`, etc. directories contain real prediction results from the AlphaFold3 examples, showing expected output formats and confidence scores.