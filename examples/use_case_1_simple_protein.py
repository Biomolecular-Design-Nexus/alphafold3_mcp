#!/usr/bin/env python3
"""
Use Case 1: Simple Protein Structure Prediction

This script demonstrates how to predict protein structure using AlphaFold3
from just the amino acid sequence. This is the most basic use case for
structure prediction.

Usage:
    python examples/use_case_1_simple_protein.py --sequence "MDPSSPNYDK..." --name "my_protein"
    python examples/use_case_1_simple_protein.py --fasta examples/data/protein.fasta
"""

import argparse
import json
import sys
from pathlib import Path


def create_alphafold3_config(protein_name: str, sequence: str, output_dir: str = "output") -> dict:
    """
    Create AlphaFold3 JSON configuration for simple protein structure prediction.

    Args:
        protein_name: Name for the prediction job
        sequence: Single-letter amino acid sequence
        output_dir: Directory where results will be saved

    Returns:
        Dictionary containing the AF3 JSON configuration
    """
    config = {
        "name": protein_name,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": sequence
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


def main():
    parser = argparse.ArgumentParser(
        description="Predict protein structure using AlphaFold3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From sequence string
    python %(prog)s --sequence "MDPSSPNYDKWEMERT..." --name "kinase"

    # From FASTA file
    python %(prog)s --fasta examples/data/protein.fasta

    # Custom output directory
    python %(prog)s --sequence "MDPSS..." --name "test" --output my_results
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--sequence',
        help='Protein amino acid sequence (single letter codes)'
    )
    input_group.add_argument(
        '--fasta',
        help='Path to FASTA file containing single protein sequence'
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

    # Get sequence and name
    if args.fasta:
        try:
            name, sequence = read_fasta(args.fasta)
            if args.name:
                name = args.name
        except Exception as e:
            print(f"Error reading FASTA file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        sequence = args.sequence
        name = args.name or "protein_prediction"

    # Validate sequence
    if not validate_protein_sequence(sequence):
        print("Error: Invalid amino acid sequence. Use single letter codes (A-Z).", file=sys.stderr)
        sys.exit(1)

    print(f"Protein name: {name}")
    print(f"Sequence length: {len(sequence)} amino acids")
    print(f"Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")

    # Create configuration
    config = create_alphafold3_config(name, sequence, args.output)

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
    print(f"   - Structure: {args.output}/{name}_model.cif")
    print(f"   - Confidences: {args.output}/{name}_confidences.json")


if __name__ == "__main__":
    main()