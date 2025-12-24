# Configuration Files

Configuration files for MCP scripts, extracted from use case parameters.

## Files

| Config File | Description | Used By |
|-------------|-------------|---------|
| `default_config.json` | Default settings for all scripts | All scripts |
| `simple_protein_config.json` | Simple protein structure prediction | `simple_protein_prediction.py` |
| `protein_ligand_config.json` | Protein-ligand complex prediction | `protein_ligand_complex.py` |
| `precomputed_msa_config.json` | Pre-computed MSA prediction | `precomputed_msa_prediction.py` |
| `batch_variants_config.json` | Batch variant prediction | `batch_variants_prediction.py` |

## Usage

### Using Default Config
Scripts use built-in defaults when no config is specified:
```bash
python scripts/clean/simple_protein_prediction.py --sequence "MDPSS..." --output config.json
```

### Using Custom Config
```bash
python scripts/clean/simple_protein_prediction.py --sequence "MDPSS..." --config configs/simple_protein_config.json --output config.json
```

### Override Specific Parameters
```bash
# Override model seeds
python scripts/clean/simple_protein_prediction.py --sequence "MDPSS..." --seeds 1 2 3 --output config.json
```

## Config Structure

### Common Fields

All configs contain these common sections:

```json
{
  "_description": "Configuration description",
  "_source": "Source use case file",

  "model": {
    "seeds": [1],
    "dialect": "alphafold3",
    "version": 1
  },

  "output": {
    "format": "json",
    "include_metadata": true
  },

  "validation": {
    "check_sequence": true,
    "valid_amino_acids": "ACDEFGHIKLMNPQRSTVWY"
  }
}
```

### Script-Specific Fields

#### Simple Protein Config
```json
{
  "naming": {
    "default_name": "protein_prediction",
    "name_format": "{protein_name}_prediction"
  }
}
```

#### Protein-Ligand Config
```json
{
  "chains": {
    "protein": ["A"],
    "ligand": ["B"]
  },
  "ligand": {
    "supported_types": ["smiles", "ccd_id"]
  }
}
```

#### Pre-computed MSA Config
```json
{
  "msa": {
    "supported_modes": ["a3m", "msa"],
    "default_mode": "a3m",
    "path_calculation": "relative_preferred"
  }
}
```

#### Batch Variants Config
```json
{
  "variants": {
    "max_variants": null,
    "supported_input_types": ["fasta", "mutations"],
    "mutation_format": "A123V"
  },
  "naming": {
    "variant_dir_prefix": "seq"
  }
}
```

## Config Parameters Reference

### Model Parameters
- `seeds`: List of model seeds for prediction (default: `[1]`)
- `dialect`: AlphaFold3 dialect (always `"alphafold3"`)
- `version`: Configuration version (always `1`)

### Naming Parameters
- `default_name`: Default name when not specified
- `name_format`: Format string for generated names
- `variant_dir_prefix`: Prefix for variant subdirectories

### Validation Parameters
- `check_sequence`: Enable protein sequence validation
- `valid_amino_acids`: Valid amino acid characters
- `min_sequence_length`: Minimum sequence length (default: 1)
- `max_sequence_length`: Maximum sequence length (default: 10000)

### Output Parameters
- `format`: Output format (always `"json"`)
- `indent`: JSON indentation (default: 2)
- `include_metadata`: Include execution metadata

### Path Parameters
- `model_dir`: AlphaFold3 model directory
- `db_dir`: AlphaFold3 database directory
- `output_dir`: Default output directory
- `temp_dir`: Temporary files directory

## Customizing Configs

### Create Custom Config
1. Copy an existing config file
2. Modify the parameters you want to change
3. Use with `--config` parameter

Example custom config:
```json
{
  "_description": "Custom protein prediction config",

  "model": {
    "seeds": [1, 2, 3, 4, 5],
    "dialect": "alphafold3",
    "version": 1
  },

  "naming": {
    "default_name": "my_protein",
    "name_format": "{protein_name}_custom"
  },

  "validation": {
    "min_sequence_length": 10,
    "max_sequence_length": 5000
  }
}
```

### Config Precedence
1. Command line arguments (highest priority)
2. Custom config file (specified with `--config`)
3. Script defaults (lowest priority)

Example:
```bash
# This will use seeds [7, 8, 9] (from command line)
# even if config file specifies different seeds
python script.py --config custom.json --seeds 7 8 9
```