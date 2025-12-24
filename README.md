# AlphaFold3 MCP

> AI-powered protein structure prediction and variant analysis using AlphaFold3 through Model Context Protocol (MCP)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The AlphaFold3 MCP provides seamless integration between AI assistants (Claude Code, Gemini CLI) and AlphaFold3 structure prediction capabilities. This enables conversational protein modeling workflows where you can simply describe what you want to predict and get results.

### Features
- **Fast sync tools** for configuration and validation (< 10 seconds)
- **Async job management** for long-running predictions (minutes to hours)
- **Batch variant processing** for protein engineering workflows
- **Pre-computed MSA support** for accelerated predictions
- **Protein-ligand complex modeling** with SMILES and CCD support
- **Comprehensive job tracking** with logs and status monitoring

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   ├── server.py           # MCP server (13 tools)
│   └── jobs/               # Job management system
├── scripts/
│   ├── clean/              # Standalone scripts
│   │   ├── simple_protein_prediction.py      # Basic protein prediction
│   │   ├── protein_ligand_complex.py         # Protein-ligand complexes
│   │   ├── precomputed_msa_prediction.py     # MSA-based prediction
│   │   └── batch_variants_prediction.py      # Batch variant processing
│   └── lib/                # Shared utilities
├── examples/
│   └── data/               # Demo data
├── configs/                # Configuration files
└── jobs/                   # Job queue and results
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.11+
- 8GB+ RAM (16GB+ recommended for larger proteins)
- GPU support optional but recommended for production use

### Create Environment
Please strictly follow the information in `reports/step3_environment.md` for the complete setup procedure. An example workflow is shown below.

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/alphafold3_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.11 -y
# or: conda create -p ./env python=3.11 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install Dependencies
pip install -r requirements.txt

# Install MCP dependencies
pip install fastmcp loguru --ignore-installed
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/clean/simple_protein_prediction.py` | Create AF3 config for basic protein prediction | See below |
| `scripts/clean/protein_ligand_complex.py` | Create AF3 config for protein-ligand complex | See below |
| `scripts/clean/precomputed_msa_prediction.py` | Create AF3 config with pre-computed MSA | See below |
| `scripts/clean/batch_variants_prediction.py` | Create AF3 configs for batch variants | See below |

### Script Examples

#### Simple Protein Prediction

```bash
# Activate environment
mamba activate ./env

# Run script
python scripts/clean/simple_protein_prediction.py \
  --input examples/data/sample_protein.fasta \
  --output results/simple_config.json \
  --name "kinase_test"
```

**Parameters:**
- `--input, -i`: Protein FASTA file or sequence string (required)
- `--sequence`: Direct amino acid sequence input (alternative to --input)
- `--output, -o`: Output JSON configuration file (default: input.json)
- `--name`: Protein name for prediction (optional)
- `--config`: Config file override (optional)

#### Protein-Ligand Complex

```bash
python scripts/clean/protein_ligand_complex.py \
  --protein-file examples/data/sample_protein.fasta \
  --smiles "CCO" \
  --name "ethanol_complex" \
  --output complex_config.json
```

**Parameters:**
- `--protein-file`: Protein FASTA file (required if no --protein-seq)
- `--protein-seq`: Direct amino acid sequence (alternative)
- `--smiles`: SMILES string for ligand (optional)
- `--ccd-id`: Chemical Component Dictionary ID (alternative to SMILES)
- `--complex-name`: Complex name (optional)

#### Pre-computed MSA Prediction

```bash
python scripts/clean/precomputed_msa_prediction.py \
  --protein examples/data/wt.fasta \
  --msa examples/data/wt.a3m \
  --mode a3m \
  --output msa_config.json
```

**Parameters:**
- `--protein`: Protein FASTA file (required if no --sequence)
- `--sequence`: Direct amino acid sequence (alternative)
- `--msa`: Pre-computed MSA file in A3M format (required)
- `--mode`: Prediction mode: "a3m" or "msa" (default: a3m)

#### Batch Variants Processing

```bash
python scripts/clean/batch_variants_prediction.py \
  --fasta examples/data/protein_variants.fasta \
  --output-dir batch_results \
  --max-variants 50
```

**Parameters:**
- `--fasta`: Multi-FASTA file with variants (required if no --variants)
- `--variants`: Text file with mutation list (alternative)
- `--template`: Template FASTA for mutations (required with --variants)
- `--output-dir`: Output directory for variant configs (default: batch_output)
- `--max-variants`: Limit number of variants processed (optional)

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name alphafold3
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add alphafold3 -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "alphafold3": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/alphafold3_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/alphafold3_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from alphafold3?
```

#### Basic Usage
```
Use create_simple_protein_config with sequence "MDPSSPNYDKWEMER" and name "test_protein"
```

#### With File References
```
Validate the sequences in @examples/data/sample_protein.fasta
```

#### Long-Running Tasks (Submit API)
```
Submit structure prediction for @examples/1iep_a3m_fix
Then check the job status
```

#### Batch Processing
```
Prepare variants from @examples/data/protein_variants.fasta using @examples/1iep_a3m_fix/1iep_a3m_fix_data.json
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/sample_protein.fasta` | Reference a specific file |
| `@configs/default_config.json` | Reference a config file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "alphafold3": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/alphafold3_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/alphafold3_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use create_simple_protein_config with sequence "MDPSSPNYDKWEMER"
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `create_simple_protein_config` | Create basic protein config | sequence, name, output_dir |
| `validate_fasta_sequences` | Validate protein sequences | fasta_path |
| `prepare_variants` | Prepare variant configurations | variants_fasta, wt_data_json, output_dir |
| `get_server_info` | Get server capabilities | None |
| `get_example_workflows` | Get workflow documentation | None |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_structure_prediction` | AlphaFold3 structure prediction | data_path, device, model_dir, db_dir |
| `submit_batch_variants` | Batch variant processing | input_dir, device, skip_existing |
| `submit_prepare_and_predict_variants` | End-to-end variant workflow | variants_fasta, wt_data_json, output_dir |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs |

---

## Examples

### Example 1: Simple Protein Structure Prediction

**Goal:** Predict structure of a single protein from sequence

**Using Script:**
```bash
python scripts/clean/simple_protein_prediction.py \
  --input examples/data/sample_protein.fasta \
  --output results/kinase_config.json \
  --name "kinase_test"
```

**Using MCP (in Claude Code):**
```
Create a simple protein config for the sequence in @examples/data/sample_protein.fasta with name "kinase_test"
```

**Expected Output:**
- `results/kinase_config.json` - AlphaFold3 input configuration
- JSON structure with protein sequence, model seeds, and AF3 dialect

### Example 2: Protein-Ligand Complex Prediction

**Goal:** Model protein bound to small molecule

**Using Script:**
```bash
python scripts/clean/protein_ligand_complex.py \
  --protein-file examples/data/sample_protein.fasta \
  --smiles "CCO" \
  --name "ethanol_complex" \
  --output complex_config.json
```

**Using MCP (in Claude Code):**
```
Create a protein-ligand complex config using @examples/data/sample_protein.fasta and SMILES "CCO" for ethanol binding
```

### Example 3: Batch Variant Processing

**Goal:** Process multiple protein variants at once

**Using Script:**
```bash
python scripts/clean/batch_variants_prediction.py \
  --fasta examples/data/protein_variants.fasta \
  --output-dir batch_results
```

**Using MCP (in Claude Code):**
```
Prepare batch variants from @examples/data/protein_variants.fasta and save configs to batch_results/
```

### Example 4: Full AlphaFold3 Prediction Workflow

**Using MCP (in Claude Code):**
```
1. Submit structure prediction for @examples/1iep_a3m_fix
2. Monitor the job with get_job_status
3. When complete, get results with get_job_result
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `sample_protein.fasta` | Sample kinase sequence (289 AA) | simple_protein_prediction |
| `protein_variants.fasta` | 3 kinase variants (WT + 2 mutants) | batch_variants_prediction |
| `wt.fasta` | Subtilisin wild-type (288 AA) | precomputed_msa_prediction |
| `wt.a3m` | Pre-computed MSA (~3MB, 8,424 sequences) | precomputed_msa_prediction |
| `sample_mutations.txt` | Example mutation list (5 mutations) | batch_variants_prediction |
| `subtilisin_variants.fasta` | 116 subtilisin variants | batch_variants_prediction |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `default_config.json` | Default settings | model, paths, validation, logging |
| `simple_protein_config.json` | Simple protein settings | naming, validation rules |
| `protein_ligand_config.json` | Complex settings | chain IDs, ligand types, validation |
| `precomputed_msa_config.json` | MSA settings | MSA modes, path calculation |
| `batch_variants_config.json` | Batch settings | variant handling, mutations, naming |

### Config Example

```json
{
  "model": "alphafold3",
  "default_name": "protein_prediction",
  "validation": {
    "min_sequence_length": 10,
    "max_sequence_length": 2000,
    "allowed_amino_acids": "ACDEFGHIKLMNPQRSTVWY"
  },
  "paths": {
    "output_dir": "output",
    "relative_paths": true
  }
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.11 -y
mamba activate ./env
pip install -r requirements.txt
```

**Problem:** Import errors
```bash
# Verify installation
python -c "from src.server import mcp; print('OK')"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove alphafold3
claude mcp add alphafold3 -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
python -c "
from src.server import mcp
print(list(mcp.list_tools().keys()))
"
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job log
cat jobs/<job_id>/job.log
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details
```

### File Path Issues

**Problem:** File not found errors
- Always use absolute paths in production
- Ensure demo data exists: `ls examples/data/`
- Check file permissions: `ls -la examples/data/`

**Problem:** Permission denied
```bash
# Fix permissions
chmod +r examples/data/*
```

### Performance Issues

**Problem:** Slow predictions
- Use pre-computed MSAs when available (`examples/data/wt.a3m`)
- Enable GPU acceleration with `device=0`
- Use batch processing for multiple variants

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test individual scripts
python scripts/clean/simple_protein_prediction.py --sequence "MDPSSPNYDKWEMER" --output test.json

# Test MCP server
python src/tests/test_simple.py
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
fastmcp dev src/server.py
```

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export PYTHONUNBUFFERED=1
```

---

## Performance Characteristics

### Script Operations (Local)
- **Config creation**: < 1 second for any size protein
- **Validation**: < 1 second for 1000+ sequences
- **Variant preparation**: 1-5 seconds for 100 variants

### AlphaFold3 Predictions (MCP Submit API)
- **Small protein** (50-100 AA): 15-30 minutes
- **Medium protein** (100-300 AA): 30-60 minutes
- **Large protein** (300+ AA): 1-2 hours
- **Batch variants** (10 variants): 3-6 hours
- **Large batch** (100+ variants): 10-24+ hours

**Note**: Actual runtimes depend on GPU, protein complexity, MSA availability, and system load.

---

## License

MIT License

## Credits

Based on [AlphaFold3](https://github.com/google-deepmind/alphafold3) by Google DeepMind