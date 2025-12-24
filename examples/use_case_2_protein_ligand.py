#!/usr/bin/env python3
"""
Use Case 2: Protein-Ligand Complex Prediction

This script demonstrates how to predict protein-ligand complex structures
using AlphaFold3. It supports both SMILES strings and Chemical Component
Dictionary (CCD) IDs for small molecule ligands.

Usage:
    python examples/use_case_2_protein_ligand.py --protein "MDPSS..." --ligand "CCO" --name "complex"
    python examples/use_case_2_protein_ligand.py --fasta protein.fasta --smiles "c1ccccc1" --name "benzene_complex"
    python examples/use_case_2_protein_ligand.py --fasta protein.fasta --ccd-id "ATP" --name "atp_complex"
"""

import argparse
import json
import sys
from pathlib import Path


def create_alphafold3_complex_config(
    protein_name: str,
    protein_sequence: str,
    ligand_data: str,
    ligand_type: str = "smiles",
    output_dir: str = "output"
) -> dict:
    """
    Create AlphaFold3 JSON configuration for protein-ligand complex prediction.

    Args:
        protein_name: Name for the prediction job
        protein_sequence: Single-letter amino acid sequence
        ligand_data: SMILES string or CCD ID for the ligand
        ligand_type: Type of ligand specification ("smiles" or "ccd_id")
        output_dir: Directory where results will be saved

    Returns:
        Dictionary containing the AF3 JSON configuration
    """
    # Create ligand specification
    if ligand_type == "smiles":
        ligand_spec = {"smiles": ligand_data}
    elif ligand_type == "ccd_id":
        ligand_spec = {"ccd_id": ligand_data}
    else:
        raise ValueError(f"Invalid ligand type: {ligand_type}")

    config = {
        "name": protein_name,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": protein_sequence
                }
            },
            {
                "ligand": {
                    "id": ["B"],
                    **ligand_spec
                }
            }
        ],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1
    }
    return config


def read_fasta(fasta_path: str) -> tuple[str, str]:
    """
    Read a single-sequence FASTA file.

    Args:
        fasta_path: Path to the FASTA file

    Returns:
        Tuple of (sequence_name, sequence)
    """
    with open(fasta_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Empty FASTA file")

    if not lines[0].startswith('>'):
        raise ValueError("Invalid FASTA format - missing header")

    name = lines[0][1:]  # Remove '>' prefix
    sequence = ''.join(lines[1:])  # Join all sequence lines

    return name, sequence


def validate_protein_sequence(sequence: str) -> bool:
    """
    Validate that the sequence contains only valid amino acids.

    Args:
        sequence: Amino acid sequence

    Returns:
        True if valid, False otherwise
    """
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa.upper() in valid_aa for aa in sequence)


def validate_smiles(smiles: str) -> bool:
    """
    Basic validation of SMILES string (check for common characters).

    Args:
        smiles: SMILES string

    Returns:
        True if appears valid, False otherwise
    """
    valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]@+=\\/#-')
    return len(smiles) > 0 and all(c in valid_chars for c in smiles)


def main():
    parser = argparse.ArgumentParser(
        description="Predict protein-ligand complex structure using AlphaFold3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simple alcohol ligand
    python %(prog)s --protein "MDPSS..." --smiles "CCO" --name "ethanol_complex"

    # Protein from FASTA with benzene ligand
    python %(prog)s --fasta protein.fasta --smiles "c1ccccc1" --name "benzene_complex"

    # ATP ligand using CCD ID
    python %(prog)s --fasta protein.fasta --ccd-id "ATP" --name "atp_complex"

    # Example from 1IEP structure
    python %(prog)s --fasta examples/data/1iep_protein.fasta \\
        --smiles "Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1" \\
        --name "1iep_complex"
        """
    )

    # Protein input (required)
    protein_group = parser.add_mutually_exclusive_group(required=True)
    protein_group.add_argument(
        '--protein',
        help='Protein amino acid sequence (single letter codes)'
    )
    protein_group.add_argument(
        '--fasta',
        help='Path to FASTA file containing protein sequence'
    )

    # Ligand input (required, mutually exclusive)
    ligand_group = parser.add_mutually_exclusive_group(required=True)
    ligand_group.add_argument(
        '--smiles',
        help='SMILES string for small molecule ligand'
    )
    ligand_group.add_argument(
        '--ccd-id',
        help='Chemical Component Dictionary ID (e.g., ATP, NAD, HEM)'
    )
    ligand_group.add_argument(
        '--ligand',
        help='Legacy alias for --smiles (deprecated)'
    )

    # Optional parameters
    parser.add_argument(
        '--name',
        help='Name for the prediction (default: derived from input)'
    )
    parser.add_argument(
        '--output',
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Only create JSON config file, do not run prediction'
    )

    args = parser.parse_args()

    # Get protein sequence and name
    if args.fasta:
        try:
            protein_name, protein_sequence = read_fasta(args.fasta)
        except Exception as e:
            print(f"Error reading FASTA file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        protein_sequence = args.protein
        protein_name = "protein"

    # Override name if provided
    if args.name:
        protein_name = args.name

    # Get ligand specification
    if args.smiles or args.ligand:
        ligand_data = args.smiles or args.ligand  # Support legacy --ligand
        ligand_type = "smiles"
        ligand_desc = f"SMILES: {ligand_data}"
    else:  # args.ccd_id
        ligand_data = getattr(args, 'ccd_id')
        ligand_type = "ccd_id"
        ligand_desc = f"CCD ID: {ligand_data}"

    # Validate inputs
    if not validate_protein_sequence(protein_sequence):
        print("Error: Invalid protein sequence. Use single letter amino acid codes.", file=sys.stderr)
        sys.exit(1)

    if ligand_type == "smiles" and not validate_smiles(ligand_data):
        print("Error: Invalid SMILES string format.", file=sys.stderr)
        sys.exit(1)

    print(f"Complex name: {protein_name}")
    print(f"Protein length: {len(protein_sequence)} amino acids")
    print(f"Protein sequence: {protein_sequence[:50]}{'...' if len(protein_sequence) > 50 else ''}")
    print(f"Ligand: {ligand_desc}")

    # Create configuration
    config = create_alphafold3_complex_config(
        protein_name, protein_sequence, ligand_data, ligand_type, args.output
    )

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)

    # Write JSON configuration
    json_path = output_path / "input.json"
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {json_path}")

    if args.json_only:
        print("JSON configuration created. Use --json-only=false to run prediction.")
        return

    # Instructions for running AlphaFold3
    print("\nTo run AlphaFold3 prediction:")
    print("1. Activate the conda environment:")
    print("   mamba activate ./env")
    print()
    print("2. Run AlphaFold3:")
    print(f"   python src/tools/af3_predict_structure.py \\")
    print(f"     --json_path={json_path} \\")
    print(f"     --model_dir=repo/alphafold3/model \\")
    print(f"     --db_dir=repo/alphafold3/alphafold3_db \\")
    print(f"     --output_dir={args.output}")
    print()
    print("3. Results will be saved in the output directory:")
    print(f"   - Complex structure: {args.output}/{protein_name}_model.cif")
    print(f"   - Confidence scores: {args.output}/{protein_name}_confidences.json")
    print()
    print("Note: Protein-ligand predictions may take longer than protein-only predictions")
    print("      due to additional sampling requirements for ligand conformations.")


if __name__ == "__main__":
    main()