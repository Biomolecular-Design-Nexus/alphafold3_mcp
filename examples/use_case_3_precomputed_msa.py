#!/usr/bin/env python3
"""
Use Case 3: Protein Structure Prediction with Pre-computed MSA

This script demonstrates how to predict protein structure using AlphaFold3
with a pre-computed Multiple Sequence Alignment (A3M format). This approach
is faster as it skips the MSA generation step and can be used for iterative
design workflows or when you have a high-quality MSA.

Usage:
    python examples/use_case_3_precomputed_msa.py --fasta protein.fasta --msa alignment.a3m --name "fast_prediction"
    python examples/use_case_3_precomputed_msa.py --protein "MDPSS..." --msa alignment.a3m --mode a3m
"""

import argparse
import json
import sys
from pathlib import Path


def create_alphafold3_msa_config(
    protein_name: str,
    protein_sequence: str,
    msa_path: str,
    mode: str = "a3m",
    output_dir: str = "output"
) -> dict:
    """
    Create AlphaFold3 JSON configuration for prediction with pre-computed MSA.

    Args:
        protein_name: Name for the prediction job
        protein_sequence: Single-letter amino acid sequence
        msa_path: Path to the A3M MSA file (relative to JSON location)
        mode: Prediction mode ("a3m" or "msa")
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
                    "sequence": protein_sequence,
                    "msa_path": msa_path
                }
            }
        ],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1
    }

    # Add mode-specific parameters
    if mode == "msa":
        # MSA mode: skip template search and MSA generation
        config["mode"] = "msa"

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


def validate_a3m_file(a3m_path: str) -> tuple[bool, str]:
    """
    Validate that the A3M file exists and has basic A3M format.

    Args:
        a3m_path: Path to the A3M file

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(a3m_path)

    if not path.exists():
        return False, f"A3M file not found: {a3m_path}"

    try:
        with open(path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith('>'):
                return False, "A3M file must start with FASTA header line (>)"
        return True, ""
    except Exception as e:
        return False, f"Error reading A3M file: {e}"


def count_sequences_in_a3m(a3m_path: str) -> int:
    """
    Count the number of sequences in the A3M file.

    Args:
        a3m_path: Path to the A3M file

    Returns:
        Number of sequences in the alignment
    """
    count = 0
    with open(a3m_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Predict protein structure using pre-computed MSA with AlphaFold3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prediction Modes:
    a3m: Use pre-computed A3M MSA + optional template search (default)
    msa: Use pre-computed MSA/templates, inference only (fastest for variants)

Examples:
    # Basic A3M mode
    python %(prog)s --fasta protein.fasta --msa alignment.a3m --name "fast_pred"

    # MSA mode (fastest, no template search)
    python %(prog)s --protein "MDPSS..." --msa alignment.a3m --mode msa

    # Using existing example
    python %(prog)s --fasta examples/subtilisin/wt.fasta \\
        --msa examples/subtilisin/wt/wt.a3m --name "subtilisin_fast"
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

    # MSA input (required)
    parser.add_argument(
        '--msa',
        required=True,
        help='Path to A3M MSA file'
    )

    # Optional parameters
    parser.add_argument(
        '--name',
        help='Name for the prediction (default: derived from input)'
    )
    parser.add_argument(
        '--mode',
        choices=['a3m', 'msa'],
        default='a3m',
        help='Prediction mode (default: a3m)'
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
        protein_name = "protein_msa"

    # Override name if provided
    if args.name:
        protein_name = args.name

    # Validate inputs
    if not validate_protein_sequence(protein_sequence):
        print("Error: Invalid protein sequence. Use single letter amino acid codes.", file=sys.stderr)
        sys.exit(1)

    # Validate A3M file
    is_valid, error_msg = validate_a3m_file(args.msa)
    if not is_valid:
        print(f"Error: {error_msg}", file=sys.stderr)
        sys.exit(1)

    # Count sequences in MSA
    try:
        seq_count = count_sequences_in_a3m(args.msa)
    except Exception as e:
        print(f"Warning: Could not count sequences in A3M file: {e}", file=sys.stderr)
        seq_count = "unknown"

    print(f"Protein name: {protein_name}")
    print(f"Sequence length: {len(protein_sequence)} amino acids")
    print(f"Sequence: {protein_sequence[:50]}{'...' if len(protein_sequence) > 50 else ''}")
    print(f"MSA file: {args.msa}")
    print(f"MSA sequences: {seq_count}")
    print(f"Prediction mode: {args.mode}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)

    # Calculate relative path from output directory to MSA file
    msa_abs_path = Path(args.msa).resolve()
    output_abs_path = output_path.resolve()
    try:
        msa_relative = msa_abs_path.relative_to(output_abs_path)
        msa_path_in_json = str(msa_relative)
    except ValueError:
        # Files are not on the same relative path, use absolute path
        msa_path_in_json = str(msa_abs_path)

    print(f"MSA path in JSON: {msa_path_in_json}")

    # Create configuration
    config = create_alphafold3_msa_config(
        protein_name, protein_sequence, msa_path_in_json, args.mode, args.output
    )

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
    if args.mode == "a3m":
        print(f"     --mode=a3m")
    elif args.mode == "msa":
        print(f"     --mode=msa")
    print()
    print("3. Results will be saved in the output directory:")
    print(f"   - Structure: {args.output}/{protein_name}_model.cif")
    print(f"   - Confidences: {args.output}/{protein_name}_confidences.json")
    print()
    if args.mode == "a3m":
        print("Note: A3M mode will perform template search and inference (moderate speed).")
    else:
        print("Note: MSA mode performs only inference (fastest, good for variants).")


if __name__ == "__main__":
    main()