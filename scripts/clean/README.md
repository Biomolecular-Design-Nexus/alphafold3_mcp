# Clean MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported (Python standard library only)
2. **Self-Contained**: Functions inlined where possible, no repo dependencies
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Dependencies | Config |
|--------|-------------|--------------|--------|
| `simple_protein_prediction.py` | Create AlphaFold3 config for protein structure prediction | Standard library | `configs/simple_protein_config.json` |
| `protein_ligand_complex.py` | Create AlphaFold3 config for protein-ligand complex | Standard library | `configs/protein_ligand_config.json` |
| `precomputed_msa_prediction.py` | Create AlphaFold3 config with pre-computed MSA | Standard library | `configs/precomputed_msa_config.json` |
| `batch_variants_prediction.py` | Create AlphaFold3 configs for batch variants | Standard library | `configs/batch_variants_config.json` |

## Usage

```bash
# Simple protein prediction from sequence
python scripts/clean/simple_protein_prediction.py --sequence "MDPSSPNYD..." --output config.json

# Simple protein prediction from FASTA
python scripts/clean/simple_protein_prediction.py --input protein.fasta --output config.json

# Protein-ligand complex with SMILES
python scripts/clean/protein_ligand_complex.py --protein-file protein.fasta --smiles "CCO" --output complex.json

# Protein-ligand complex with CCD ID
python scripts/clean/protein_ligand_complex.py --protein-seq "MDPSS..." --ccd-id "ATP" --output complex.json

# Pre-computed MSA (A3M mode)
python scripts/clean/precomputed_msa_prediction.py --protein protein.fasta --msa alignment.a3m --output msa.json

# Pre-computed MSA (MSA mode - fastest)
python scripts/clean/precomputed_msa_prediction.py --protein protein.fasta --msa alignment.a3m --mode msa --output msa.json

# Batch variants from multi-FASTA
python scripts/clean/batch_variants_prediction.py --fasta variants.fasta --output-dir batch_results

# Batch variants from mutation list
python scripts/clean/batch_variants_prediction.py --variants mutations.txt --template template.fasta --output-dir batch_results
```

## Using with Custom Config Files

```bash
# Use custom configuration
python scripts/clean/simple_protein_prediction.py --sequence "MDPSS..." --config my_config.json --output result.json
```

Example custom config:
```json
{
  "model_seeds": [1, 2, 3],
  "default_name": "my_protein"
}
```

## Shared Library

Common functions are in `scripts/lib/`:
- `io.py`: File loading/saving (`read_fasta`, `save_json`, `load_json`)
- `validation.py`: Input validation (`validate_protein_sequence`, `validate_smiles`, `validate_a3m_file`)
- `utils.py`: General utilities (`count_sequences_in_a3m`, `calculate_relative_path`)

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
# Import the function
from scripts.clean.simple_protein_prediction import run_simple_protein_prediction

# In MCP tool:
@mcp.tool()
def predict_protein_structure(
    input_file: str = None,
    sequence: str = None,
    protein_name: str = None,
    output_file: str = None
):
    \"\"\"Create AlphaFold3 configuration for protein structure prediction.\"\"\"
    return run_simple_protein_prediction(
        input_file=input_file,
        sequence=sequence,
        protein_name=protein_name,
        output_file=output_file
    )
```

## Function Signatures

### `run_simple_protein_prediction`
```python
def run_simple_protein_prediction(
    input_file: Optional[Union[str, Path]] = None,
    sequence: Optional[str] = None,
    protein_name: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

### `run_protein_ligand_complex`
```python
def run_protein_ligand_complex(
    protein_file: Optional[Union[str, Path]] = None,
    protein_sequence: Optional[str] = None,
    smiles: Optional[str] = None,
    ccd_id: Optional[str] = None,
    complex_name: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

### `run_precomputed_msa_prediction`
```python
def run_precomputed_msa_prediction(
    protein_file: Optional[Union[str, Path]] = None,
    protein_sequence: Optional[str] = None,
    msa_file: Union[str, Path] = None,
    mode: str = "a3m",
    protein_name: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

### `run_batch_variants_prediction`
```python
def run_batch_variants_prediction(
    fasta_file: Optional[Union[str, Path]] = None,
    variants_file: Optional[Union[str, Path]] = None,
    template_file: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "batch_output",
    max_variants: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]
```

## Output Format

All functions return a dictionary with:
```python
{
    "config": dict,          # Generated AlphaFold3 configuration
    "output_file": str,      # Path to output file (if saved)
    "metadata": dict,        # Execution metadata
    # Additional fields depending on script
}
```

## Testing

The scripts have been tested with real data:
- ✅ Simple protein prediction with sequences and FASTA files
- ✅ Protein-ligand complex with SMILES and CCD IDs
- ✅ Pre-computed MSA with actual MSA files (8424 sequences)
- ✅ Batch variants with multi-FASTA files and mutation lists

Test outputs are generated in `test_output/` directory.