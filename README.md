# AlphaFold3 MCP service

## Overview
This repository creates MCP service for [AlphaFold3](https://github.com/google-deepmind/alphafold3). It supports basic AlpaFold3 structure prediction funcations, and added two functions:
1. Run AlphaFold3 with predefined a3m files;
2. Support AlpaFold3 config preparation for protein variants with the same a3m file (different query sequence);
3. Support batch structure prediction.

## Installation
```shell
mamba env create -f environment.yml -p ./env -y
mamba activate ./env

pip install -r requirements.txt
pip install scikit-build-core cmake ninja loguru 
pip install --ignore-installed fastmcp

# Install AlphaFold3
mkdir repo
cd repo
git clone https://github.com/charlesxu90/alphafold3
cd alphafold3
pip install -e .

# Building data
build_data

export AF3_REPO_PATH=$(pwd)
# Install model
# Obtain license from Google DeepMind and put it into repo/alphafold/model

# Install Jackhmmer (add seq_limit to reduce RAM usage)
mkdir /tmp/hmmer_build
wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /tmp/hmmer_build
cd /tmp/hmmer_build && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check
tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz
cp $AF3_REPO_PATH/docker/jackhmmer_seq_limit.patch /tmp/hmmer_build
patch -p0 < jackhmmer_seq_limit.patch
cd hmmer-3.4 
./configure --prefix=$CONDA_PREFIX
make -j
make install
cd easel && make install
rm -R /tmp/hmmer_build

# Download AlphaFold3_database
# cd $AF3_REPO_PATH
# bash fetch_databases.sh ./alphafold3_db
# or soft link to existing downloads: ln -s /mnt/data/data_repository/bio-seq/af3_db/ alphafold3_db
```

## Local usage
### 1. Run AlphaFold3 with default config file
```shell
python scripts/alphafold3_runner.py default examples/1iep
```

### 2. Run AlphaFold3 with prepared a3m file
```shell
python scripts/alphafold3_runner.py a3m examples/1iep_a3m_fix
```

### 3. Prepare AlphaFold3 variant configs for batch processing
```shell
python scripts/prepare_variants.py --variants-fasta examples/subtilisin/sequences.fasta \
    --wt-data-json examples/subtilisin/wt/subtilisin_wt_data.json \
    --output-dir examples/subtilisin/variants
```

### 4. Batch AlphaFold3 structure prediction with configs
```shell
python scripts/alphafold3_runner.py batch examples/subtilisin/variants
```

## MCP usage

### Debu MCP server
```shell
cd tool-mcps/alphafold3_mcp
mamba activate ./env
fastmcp run src/alphafold3_mcp.py:mcp --transport http --port 8001 --python ./env/bin/python 
```

### Install MCP server
```shell
fastmcp install claude-code tool-mcps/alphafold3_mcp/src/alphafold3_mcp.py --python tool-mcps/alphafold3_mcp/env/bin/python
fastmcp install gemini-cli tool-mcps/alphafold3_mcp/src/alphafold3_mcp.py --python tool-mcps/alphafold3_mcp/env/bin/python
```
### Call MCP service
1. Basic end to end structure prediction give sequences
```markdown
Please predict the complex of 1iep (seq: MDPSSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVSAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQ) and ligand (Smiles: Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1) using the alphafold3_mcp. save it to @examples/1iep_raw .

Please convert the relative path to absolution path before calling the MCP servers. 
```
2. Batch variant structure prediction give a wt reference result
```markdown
Please predict the structure of variants in @examples/subtilisin/sequences.fasta with reference to the wild-type config @examples/subtilisin/wt/subtilisin_wt_data.json and save the predicted results in @examples/subtilisin/variants using alphafold3_mcp.

Please convert the relative path to absolution path before calling the MCP servers. 
```
