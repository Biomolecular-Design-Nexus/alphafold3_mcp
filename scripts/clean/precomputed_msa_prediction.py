#!/usr/bin/env python3
"""
Script: precomputed_msa_prediction.py
Description: Create AlphaFold3 configuration for prediction with pre-computed MSA

Original Use Case: examples/use_case_3_precomputed_msa.py
Dependencies Removed: None (only uses Python standard library)

Usage:
    python scripts/clean/precomputed_msa_prediction.py --protein protein.fasta --msa alignment.a3m --output config.json
    python scripts/clean/precomputed_msa_prediction.py --sequence "MDPSS..." --msa alignment.a3m --mode msa --output config.json

Example:
    python scripts/clean/precomputed_msa_prediction.py --protein examples/subtilisin/wt.fasta --msa examples/subtilisin/wt/wt.a3m --output fast_pred.json
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
    "default_name": "protein_msa_prediction",
    "mode": "a3m"  # Default mode: a3m (with template search) or msa (inference only)
}

# ==============================================================================
# Inlined Utility Functions (simplified from use case)
# ==============================================================================
def read_fasta(file_path: Path) -> tuple[str, str]:
    """Read a single-sequence FASTA file. Inlined from examples/use_case_3_precomputed_msa.py"""
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


def validate_a3m_file(a3m_path: Path) -> tuple[bool, str]:
    """Validate that the A3M file exists and has basic A3M format. Inlined from use case."""
    if not a3m_path.exists():
        return False, f"A3M file not found: {a3m_path}"

    try:
        with open(a3m_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith('>'):
                return False, "A3M file must start with FASTA header line (>)"
        return True, ""
    except Exception as e:
        return False, f"Error reading A3M file: {e}"


def count_sequences_in_a3m(a3m_path: Path) -> int:
    """Count the number of sequences in the A3M file. Inlined from use case."""
    count = 0
    try:
        with open(a3m_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
    except Exception:
        count = 0
    return count


def calculate_relative_path(msa_path: Path, output_dir: Path) -> str:
    """Calculate relative path from output directory to MSA file."""
    try:
        msa_abs = msa_path.resolve()
        output_abs = output_dir.resolve()
        msa_relative = msa_abs.relative_to(output_abs)
        return str(msa_relative)
    except ValueError:
        # Files are not on the same relative path, use absolute path
        return str(msa_abs)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_precomputed_msa_prediction(
    protein_file: Optional[Union[str, Path]] = None,
    protein_sequence: Optional[str] = None,
    msa_file: Union[str, Path] = None,
    mode: str = "a3m",
    protein_name: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for protein structure prediction with pre-computed MSA.

    Args:
        protein_file: Path to FASTA protein file (mutually exclusive with protein_sequence)
        protein_sequence: Protein amino acid sequence (mutually exclusive with protein_file)
        msa_file: Path to A3M MSA file (required)
        mode: Prediction mode - "a3m" (with template search) or "msa" (inference only)
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
        >>> result = run_precomputed_msa_prediction(
        ...     protein_file="protein.fasta",
        ...     msa_file="alignment.a3m",
        ...     mode="a3m",
        ...     output_file="config.json"
        ... )
        >>> print(result['config']['name'])
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate input - exactly one protein source must be provided
    if not ((protein_file is None) ^ (protein_sequence is None)):
        raise ValueError("Exactly one of 'protein_file' or 'protein_sequence' must be provided")

    if msa_file is None:
        raise ValueError("MSA file is required")

    # Validate mode
    if mode not in ["a3m", "msa"]:
        raise ValueError("Mode must be 'a3m' or 'msa'")

    # Get protein sequence and name
    if protein_file:
        protein_file = Path(protein_file)
        if not protein_file.exists():
            raise FileNotFoundError(f"Protein file not found: {protein_file}")

        name, protein_seq = read_fasta(protein_file)
        protein_name = protein_name or name
    else:
        protein_seq = protein_sequence
        protein_name = protein_name or config["default_name"]

    # Validate protein sequence
    if not validate_protein_sequence(protein_seq):
        raise ValueError("Invalid protein sequence. Use single letter amino acid codes.")

    # Validate MSA file
    msa_path = Path(msa_file)
    is_valid, error_msg = validate_a3m_file(msa_path)
    if not is_valid:
        raise ValueError(f"MSA validation failed: {error_msg}")

    # Count sequences in MSA
    seq_count = count_sequences_in_a3m(msa_path)

    # Calculate MSA path for JSON (relative if possible)
    output_dir = Path(output_file).parent if output_file else Path.cwd()
    msa_path_in_json = calculate_relative_path(msa_path, output_dir)

    # Create AlphaFold3 configuration
    af3_config = {
        "name": protein_name,
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": protein_seq,
                    "msa_path": msa_path_in_json
                }
            }
        ],
        "modelSeeds": config["model_seeds"],
        "dialect": config["dialect"],
        "version": config["version"]
    }

    # Add mode-specific parameters
    if mode == "msa":
        # MSA mode: skip template search and MSA generation
        af3_config["mode"] = "msa"

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
            "protein_name": protein_name,
            "protein_sequence_length": len(protein_seq),
            "msa_file": str(msa_path),
            "msa_sequence_count": seq_count,
            "msa_path_in_json": msa_path_in_json,
            "mode": mode,
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
    protein_group.add_argument('--protein', '-p', help='FASTA protein file path')
    protein_group.add_argument('--sequence', '-s', help='Protein amino acid sequence')

    # MSA input (required)
    parser.add_argument('--msa', '-m', required=True, help='Path to A3M MSA file')

    # Mode selection
    parser.add_argument('--mode', choices=['a3m', 'msa'], default='a3m',
                       help='Prediction mode: a3m (with template search) or msa (inference only)')

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
    result = run_precomputed_msa_prediction(
        protein_file=args.protein,
        protein_sequence=args.sequence,
        msa_file=args.msa,
        mode=args.mode,
        protein_name=args.name,
        output_file=args.output,
        config=config
    )

    # Print results
    metadata = result['metadata']
    print(f"✅ Pre-computed MSA prediction configuration created")
    print(f"   Protein: {metadata['protein_name']}")
    print(f"   Sequence length: {metadata['protein_sequence_length']} amino acids")
    print(f"   MSA file: {metadata['msa_file']}")
    print(f"   MSA sequences: {metadata['msa_sequence_count']}")
    print(f"   Mode: {metadata['mode']}")
    if result['output_file']:
        print(f"   Config saved to: {result['output_file']}")

    return result

if __name__ == '__main__':
    main()