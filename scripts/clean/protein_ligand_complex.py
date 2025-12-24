#!/usr/bin/env python3
"""
Script: protein_ligand_complex.py
Description: Create AlphaFold3 configuration for protein-ligand complex prediction

Original Use Case: examples/use_case_2_protein_ligand.py
Dependencies Removed: None (only uses Python standard library)

Usage:
    python scripts/clean/protein_ligand_complex.py --protein-file input.fasta --smiles "CCO" --output config.json
    python scripts/clean/protein_ligand_complex.py --protein-seq "MDPSS..." --ccd-id "ATP" --output config.json

Example:
    python scripts/clean/protein_ligand_complex.py --protein-file examples/subtilisin/wt.fasta --smiles "CCO" --output complex.json
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_seeds": [1],
    "dialect": "alphafold3",
    "version": 1,
    "default_name": "protein_ligand_complex"
}

# ==============================================================================
# Inlined Utility Functions (simplified from use case)
# ==============================================================================
def read_fasta(file_path: Path) -> tuple[str, str]:
    """Read a single-sequence FASTA file. Inlined from examples/use_case_2_protein_ligand.py"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Empty FASTA file")

    if not lines[0].startswith('>'):
        raise ValueError("Invalid FASTA format - missing header")

    name = lines[0][1:]  # Remove '>' prefix
    sequence = ''.join(lines[1:])  # Join all sequence lines

    return name, sequence


def validate_protein_sequence(sequence: str) -> bool:
    """Validate that the sequence contains only valid amino acids. Inlined from use case."""
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa.upper() in valid_aa for aa in sequence)


def validate_smiles(smiles: str) -> bool:
    """Basic validation of SMILES string. Inlined from use case."""
    valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]@+=\\/#-')
    return len(smiles) > 0 and all(c in valid_chars for c in smiles)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_protein_ligand_complex(
    protein_file: Optional[Union[str, Path]] = None,
    protein_sequence: Optional[str] = None,
    smiles: Optional[str] = None,
    ccd_id: Optional[str] = None,
    complex_name: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for protein-ligand complex prediction configuration.

    Args:
        protein_file: Path to FASTA protein file (mutually exclusive with protein_sequence)
        protein_sequence: Protein amino acid sequence (mutually exclusive with protein_file)
        smiles: SMILES string for ligand (mutually exclusive with ccd_id)
        ccd_id: Chemical Component Dictionary ID (mutually exclusive with smiles)
        complex_name: Name for the complex (optional, derived from input if not provided)
        output_file: Path to save output JSON config (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - config: Generated AlphaFold3 configuration
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_protein_ligand_complex(
        ...     protein_file="protein.fasta",
        ...     smiles="CCO",
        ...     output_file="complex.json"
        ... )
        >>> print(result['config']['name'])
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate input - exactly one protein and one ligand source
    if not ((protein_file is None) ^ (protein_sequence is None)):
        raise ValueError("Exactly one of 'protein_file' or 'protein_sequence' must be provided")

    if not ((smiles is None) ^ (ccd_id is None)):
        raise ValueError("Exactly one of 'smiles' or 'ccd_id' must be provided")

    # Get protein sequence and name
    if protein_file:
        protein_file = Path(protein_file)
        if not protein_file.exists():
            raise FileNotFoundError(f"Protein file not found: {protein_file}")

        protein_name, protein_seq = read_fasta(protein_file)
    else:
        protein_seq = protein_sequence
        protein_name = "protein"

    # Validate protein sequence
    if not validate_protein_sequence(protein_seq):
        raise ValueError("Invalid protein sequence. Use single letter amino acid codes.")

    # Process ligand specification
    if smiles:
        if not validate_smiles(smiles):
            raise ValueError("Invalid SMILES string format.")
        ligand_spec = {"smiles": smiles}
        ligand_type = "smiles"
        ligand_desc = f"SMILES: {smiles}"
    else:  # ccd_id
        ligand_spec = {"ccd_id": ccd_id}
        ligand_type = "ccd_id"
        ligand_desc = f"CCD ID: {ccd_id}"

    # Set complex name
    complex_name = complex_name or f"{protein_name}_{ligand_type}_complex"

    # Create AlphaFold3 configuration
    af3_config = {
        "name": complex_name,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": protein_seq
                }
            },
            {
                "ligand": {
                    "id": ["B"],
                    **ligand_spec
                }
            }
        ],
        "modelSeeds": config["model_seeds"],
        "dialect": config["dialect"],
        "version": config["version"]
    }

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        save_json(af3_config, output_path)

    return {
        "config": af3_config,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "protein_file": str(protein_file) if protein_file else None,
            "protein_sequence_length": len(protein_seq),
            "ligand_type": ligand_type,
            "ligand_description": ligand_desc,
            "complex_name": complex_name,
            "config_used": config
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Protein input options (mutually exclusive)
    protein_group = parser.add_mutually_exclusive_group(required=True)
    protein_group.add_argument('--protein-file', '-pf', help='FASTA protein file path')
    protein_group.add_argument('--protein-seq', '-ps', help='Protein amino acid sequence')

    # Ligand input options (mutually exclusive)
    ligand_group = parser.add_mutually_exclusive_group(required=True)
    ligand_group.add_argument('--smiles', '-s', help='SMILES string for ligand')
    ligand_group.add_argument('--ccd-id', '-c', help='Chemical Component Dictionary ID')

    # Output and naming options
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--name', '-n', help='Complex name for the prediction')

    # Configuration options
    parser.add_argument('--config', help='Config file (JSON)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1],
                       help='Model seeds (default: [1])')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override seeds if provided
    if args.seeds != [1]:
        config = config or {}
        config['model_seeds'] = args.seeds

    # Run
    result = run_protein_ligand_complex(
        protein_file=getattr(args, 'protein_file'),
        protein_sequence=getattr(args, 'protein_seq'),
        smiles=args.smiles,
        ccd_id=getattr(args, 'ccd_id'),
        complex_name=args.name,
        output_file=args.output,
        config=config
    )

    # Print results
    metadata = result['metadata']
    print(f"✅ Protein-ligand complex configuration created")
    print(f"   Complex: {metadata['complex_name']}")
    print(f"   Protein length: {metadata['protein_sequence_length']} amino acids")
    print(f"   Ligand: {metadata['ligand_description']}")
    if result['output_file']:
        print(f"   Config saved to: {result['output_file']}")

    return result

if __name__ == '__main__':
    main()