# AlphaFold3 MCP Server

**AI-powered protein structure prediction and variant analysis via Docker**

An MCP (Model Context Protocol) server for AlphaFold3 structure prediction with 5 core tools:
- Submit structure predictions from sequences or MSA files
- Batch process protein variants for engineering workflows
- Run end-to-end prepare-and-predict variant pipelines
- Monitor long-running prediction jobs
- Validate and prepare AlphaFold3 input configurations

## Quick Start with Docker

### Approach 1: Pull Pre-built Image from GitHub

The fastest way to get started. A pre-built Docker image is automatically published to GitHub Container Registry on every release.

```bash
# Pull the latest image
docker pull ghcr.io/macromnex/alphafold3_mcp:latest

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add alphafold3 -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` ghcr.io/macromnex/alphafold3_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support (`nvidia-docker` or Docker with NVIDIA runtime)
- Claude Code installed

That's it! The AlphaFold3 MCP server is now available in Claude Code.

---

### Approach 2: Build Docker Image Locally

Build the image yourself and install it into Claude Code. Useful for customization or offline environments.

```bash
# Clone the repository
git clone https://github.com/MacromNex/alphafold3_mcp.git
cd alphafold3_mcp

# Build the Docker image
docker build -t alphafold3_mcp:latest .

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add alphafold3 -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` alphafold3_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support
- Claude Code installed
- Git (to clone the repository)

**About the Docker Flags:**
- `-i` — Interactive mode for Claude Code
- `--rm` — Automatically remove container after exit
- `` --user `id -u`:`id -g` `` — Runs the container as your current user, so output files are owned by you (not root)
- `--gpus all` — Grants access to all available GPUs
- `--ipc=host` — Uses host IPC namespace for better performance
- `-v` — Mounts your project directory so the container can access your data

---

## Verify Installation

After adding the MCP server, you can verify it's working:

```bash
# List registered MCP servers
claude mcp list

# You should see 'alphafold3' in the output
```

In Claude Code, you can now use all 5 AlphaFold3 tools:
- `submit_structure_prediction`
- `submit_batch_variants`
- `submit_prepare_and_predict_variants`
- `get_job_status`
- `get_job_result`

---

## Next Steps

- **Detailed documentation**: See [detail.md](detail.md) for comprehensive guides on:
  - Available MCP tools and parameters
  - Local Python environment setup (alternative to Docker)
  - Example workflows and use cases
  - Configuration file formats
  - AlphaFold3 license and model weight setup

---

## Usage Examples

Once registered, you can use the AlphaFold3 tools directly in Claude Code. Here are some common workflows:

### Example 1: Structure Prediction from Sequence

```
I have a protein sequence in /path/to/protein.fasta. Can you submit an AlphaFold3 structure prediction using submit_structure_prediction and save the results to /path/to/results/?
```

### Example 2: Batch Variant Analysis

```
I have 50 protein variants in /path/to/variants.fasta and a wild-type data JSON at /path/to/wt_data.json. Can you use submit_prepare_and_predict_variants to run end-to-end structure predictions for all variants and save to /path/to/output/?
```

### Example 3: Protein-Ligand Complex

```
I want to predict the structure of my protein with a small molecule ligand. The protein is in /path/to/protein.fasta and the ligand SMILES is "CCO". Can you prepare the AlphaFold3 config and submit a structure prediction?
```

---

## Troubleshooting

**Docker not found?**
```bash
docker --version  # Install Docker if missing
```

**GPU not accessible?**
- Ensure NVIDIA Docker runtime is installed
- Check with `docker run --gpus all ubuntu nvidia-smi`

**Claude Code not found?**
```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

**AlphaFold3 license required?**
- AlphaFold3 model weights require a license from Google DeepMind
- Apply at: https://github.com/google-deepmind/alphafold3

---

## License

CC-BY-NC-SA 4.0 (Google DeepMind)
