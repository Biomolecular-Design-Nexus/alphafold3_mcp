#!/bin/bash
#===============================================================================
# AlphaFold3 MCP Quick Setup Script
#===============================================================================

mamba env create -p ./env -f environment.yml
mamba activate ./env
pip install -r requirements.txt
cd repo/alphafold3
pip install -e .

build_data

# mount alphafold3_db to repo/alphafold3/alphafold3_db
# mount alphafold3_model to repo/alphafold3/model

pip install fastmcp loguru
