#!/usr/bin/env python3
"""
Script: simple_protein_prediction.py
Description: Create AlphaFold3 configuration for simple protein structure prediction

Original Use Case: examples/use_case_1_simple_protein.py
Dependencies Removed: None (only uses Python standard library)

Usage:
    python scripts/clean/simple_protein_prediction.py --input input.fasta --output config.json
    python scripts/clean/simple_protein_prediction.py --sequence "MDPSS..." --output config.json

Example:
    python scripts/clean/simple_protein_prediction.py --input examples/subtilisin/wt.fasta --output output.json
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
    "default_name": "protein_prediction"
}

# ==============================================================================
# Inlined Utility Functions (simplified from use case)
# ==============================================================================
def read_fasta(file_path: Path) -> tuple[str, str]:
    """Read a single-sequence FASTA file. Inlined from examples/use_case_1_simple_protein.py"""
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


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_simple_protein_prediction(
    input_file: Optional[Union[str, Path]] = None,
    sequence: Optional[str] = None,
    protein_name: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for simple protein structure prediction configuration.

    Args:
        input_file: Path to FASTA input file (mutually exclusive with sequence)
        sequence: Protein amino acid sequence (mutually exclusive with input_file)
        protein_name: Name for the prediction (optional, derived from input if not provided)
        output_file: Path to save output JSON config (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - config: Generated AlphaFold3 configuration
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_simple_protein_prediction(input_file="protein.fasta", output_file="config.json")
        >>> print(result['config']['name'])
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate input - exactly one of input_file or sequence must be provided
    if not ((input_file is None) ^ (sequence is None)):
        raise ValueError("Exactly one of 'input_file' or 'sequence' must be provided")

    # Get sequence and name
    if input_file:
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        name, seq = read_fasta(input_file)
        protein_name = protein_name or name
    else:
        seq = sequence
        protein_name = protein_name or config["default_name"]

    # Validate sequence
    if not validate_protein_sequence(seq):
        raise ValueError("Invalid amino acid sequence. Use single letter codes (A-Z).")

    # Create AlphaFold3 configuration
    af3_config = {
        "name": protein_name,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": seq
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
            "input_file": str(input_file) if input_file else None,
            "sequence_length": len(seq),
            "protein_name": protein_name,
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

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', help='FASTA input file path')
    input_group.add_argument('--sequence', '-s', help='Protein amino acid sequence')

    # Output and naming options
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--name', '-n', help='Protein name for the prediction')

    # Configuration options
    parser.add_argument('--config', '-c', help='Config file (JSON)')
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
    result = run_simple_protein_prediction(
        input_file=args.input,
        sequence=args.sequence,
        protein_name=args.name,
        output_file=args.output,
        config=config
    )

    # Print results
    metadata = result['metadata']
    print(f"✅ Simple protein prediction configuration created")
    print(f"   Protein: {metadata['protein_name']}")
    print(f"   Sequence length: {metadata['sequence_length']} amino acids")
    if result['output_file']:
        print(f"   Config saved to: {result['output_file']}")

    return result

if __name__ == '__main__':
    main()