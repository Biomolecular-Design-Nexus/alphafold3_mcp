"""
Model Context Protocol (MCP) for AlphaFold3

This MCP server provides comprehensive protein structure prediction tools using AlphaFold3.
It enables researchers to run structure predictions with different modes and process multiple
protein variants efficiently.

This MCP Server contains tools extracted from the AlphaFold3 runner scripts:

1. af3_predict_structure
   - af3_predict_structure: Run AlphaFold3 structure prediction (default, a3m, or msa mode)
   - af3_predict_batch: Run batch AlphaFold3 predictions on multiple inputs
   - af3_predict_structure_from_seq: Run full pipeline from sequences (protein, DNA, RNA, ligand)

2. af3_prepare_variants
   - af3_prepare_variants: Prepare input configs for multiple protein variants
   - af3_prepare_and_predict_variants: Combined preparation and prediction workflow

Prediction Modes:
- default: Full MSA search + template search + inference (most accurate)
- a3m: Pre-computed A3M MSA + optional template search + inference
- msa: Pre-computed MSA/templates, inference only (fastest for variants)

Supported Molecule Types (for af3_predict_structure_from_seq):
- Protein: Standard amino acid sequences
- DNA: DNA sequences (A, C, G, T)
- RNA: RNA sequences (A, C, G, U)
- Ligand: Small molecules via SMILES or CCD IDs

Usage:
    # Run the MCP server
    python alphafold3_mcp.py

    # Or use with uvicorn for production
    uvicorn alphafold3_mcp:mcp --host 0.0.0.0 --port 8000
"""

from loguru import logger
from fastmcp import FastMCP

# Import tool MCPs
from tools.af3_predict_structure import af3_predict_mcp
from tools.af3_prepare_variants import af3_variants_mcp

# Server definition and mounting
mcp = FastMCP(name="alphafold3")
logger.info("Mounting af3_predict_structure tool")
mcp.mount(af3_predict_mcp)
logger.info("Mounting af3_prepare_variants tool")
mcp.mount(af3_variants_mcp)

if __name__ == "__main__":
    logger.info("Starting AlphaFold3 MCP server")
    mcp.run()
