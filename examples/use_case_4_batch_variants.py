#!/usr/bin/env python3
"""
Use Case 4: Batch Protein Variant Prediction

This script demonstrates how to prepare and predict multiple protein variants
using AlphaFold3. This is useful for protein engineering workflows where you
want to predict the effect of mutations on protein structure.

Usage:
    python examples/use_case_4_batch_variants.py --fasta variants.fasta --output variants_output
    python examples/use_case_4_batch_variants.py --variants variants.txt --template template.fasta
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def parse_fasta_variants(fasta_path: str) -> List[Dict[str, str]]:
    """
    Parse a FASTA file containing multiple protein variants.

    Args:
        fasta_path: Path to FASTA file with multiple sequences

    Returns:
        List of dictionaries with 'name' and 'sequence' keys
    """
    variants = []
    current_name = None
    current_sequence = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous variant if exists
                if current_name is not None:
                    variants.append({
                        'name': current_name,
                        'sequence': ''.join(current_sequence)
                    })
                # Start new variant
                current_name = line[1:]  # Remove '>'
                current_sequence = []
            elif line:
                current_sequence.append(line)

    # Save last variant
    if current_name is not None:
        variants.append({
            'name': current_name,
            'sequence': ''.join(current_sequence)
        })

    return variants


def parse_variant_list(variant_file: str, template_fasta: str) -> List[Dict[str, str]]:
    """
    Parse a text file with variant definitions and apply to template.

    Args:
        variant_file: Path to file with variant descriptions
        template_fasta: Path to template FASTA sequence

    Returns:
        List of dictionaries with 'name' and 'sequence' keys
    """
    # Read template sequence
    with open(template_fasta, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or not lines[0].startswith('>'):
        raise ValueError("Invalid template FASTA file")

    template_name = lines[0][1:]
    template_seq = ''.join(lines[1:])

    # Read variant definitions
    variants = []
    with open(variant_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Apply mutations to template
            variant_seq = apply_mutations(template_seq, line)
            variant_name = f"{template_name}_var_{i+1}_{line.replace(' ', '_')}"

            variants.append({
                'name': variant_name,
                'sequence': variant_seq
            })

    return variants


def apply_mutations(template_seq: str, mutation_str: str) -> str:
    """
    Apply mutations to a template sequence.

    Args:
        template_seq: Template amino acid sequence
        mutation_str: Mutation string (e.g., "A123V G456D")

    Returns:
        Mutated sequence
    """
    sequence = list(template_seq)

    for mutation in mutation_str.split():
        mutation = mutation.strip()
        if not mutation:
            continue

        # Parse mutation (e.g., A123V)
        if len(mutation) < 3:
            continue

        orig_aa = mutation[0]
        new_aa = mutation[-1]
        pos_str = mutation[1:-1]

        try:
            pos = int(pos_str) - 1  # Convert to 0-based
        except ValueError:
            print(f"Warning: Invalid mutation format: {mutation}", file=sys.stderr)
            continue

        if 0 <= pos < len(sequence):
            if sequence[pos].upper() != orig_aa.upper():
                print(f"Warning: Position {pos+1} is {sequence[pos]}, not {orig_aa}", file=sys.stderr)
            sequence[pos] = new_aa.upper()
        else:
            print(f"Warning: Position {pos+1} out of range for mutation {mutation}", file=sys.stderr)

    return ''.join(sequence)


def create_variant_configs(variants: List[Dict[str, str]], output_dir: str) -> List[str]:
    """
    Create AlphaFold3 JSON configurations for all variants.

    Args:
        variants: List of variant dictionaries
        output_dir: Base output directory

    Returns:
        List of paths to created JSON files
    """
    output_path = Path(output_dir)
    json_paths = []

    for i, variant in enumerate(variants):
        # Create variant subdirectory
        variant_dir = output_path / f"seq_{i}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Create configuration
        config = {
            "name": variant['name'],
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": variant['sequence']
                    }
                }
            ],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1
        }

        # Write JSON file
        json_path = variant_dir / "input.json"
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)

        json_paths.append(str(json_path))

    return json_paths


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
        description="Prepare batch protein variant predictions for AlphaFold3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Formats:

1. FASTA file with multiple sequences:
   >variant_1
   MDPSSPNYDKWEMERT...
   >variant_2_A123V
   MDPSSPNYDKWEMERVT...

2. Variant list file (with template):
   A123V
   G456D
   A123V G456D

Examples:
    # From multi-FASTA
    python %(prog)s --fasta variants.fasta --output variants_results

    # From variant list
    python %(prog)s --variants mutations.txt --template wild_type.fasta --output variants_results

    # Process existing subtilisin variants
    python %(prog)s --fasta examples/subtilisin/sequences.fasta --output subtilisin_batch
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--fasta',
        help='FASTA file containing multiple protein variants'
    )
    input_group.add_argument(
        '--variants',
        help='Text file with variant definitions (requires --template)'
    )

    # Template for variant list mode
    parser.add_argument(
        '--template',
        help='Template FASTA file for variant list mode'
    )

    # Required parameters
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for all variants'
    )

    # Optional parameters
    parser.add_argument(
        '--max-variants',
        type=int,
        help='Maximum number of variants to process'
    )
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Only create JSON config files, do not run predictions'
    )

    args = parser.parse_args()

    # Validate input combination
    if args.variants and not args.template:
        print("Error: --template is required when using --variants", file=sys.stderr)
        sys.exit(1)

    # Parse input variants
    try:
        if args.fasta:
            variants = parse_fasta_variants(args.fasta)
            input_desc = f"FASTA file: {args.fasta}"
        else:  # args.variants
            variants = parse_variant_list(args.variants, args.template)
            input_desc = f"Variant list: {args.variants}, Template: {args.template}"

    except Exception as e:
        print(f"Error parsing input: {e}", file=sys.stderr)
        sys.exit(1)

    if not variants:
        print("Error: No variants found in input", file=sys.stderr)
        sys.exit(1)

    # Apply max variants limit
    if args.max_variants and len(variants) > args.max_variants:
        print(f"Limiting to first {args.max_variants} variants")
        variants = variants[:args.max_variants]

    print(f"Input: {input_desc}")
    print(f"Total variants: {len(variants)}")
    print(f"Output directory: {args.output}")

    # Validate all sequences
    invalid_variants = []
    for i, variant in enumerate(variants):
        if not validate_protein_sequence(variant['sequence']):
            invalid_variants.append(f"  {i+1}: {variant['name']}")

    if invalid_variants:
        print("Error: Invalid protein sequences found:", file=sys.stderr)
        for invalid in invalid_variants:
            print(invalid, file=sys.stderr)
        sys.exit(1)

    # Show variant summary
    print("\nVariant Summary:")
    for i, variant in enumerate(variants[:5]):  # Show first 5
        print(f"  {i+1}. {variant['name']} ({len(variant['sequence'])} aa)")
    if len(variants) > 5:
        print(f"  ... and {len(variants) - 5} more variants")

    # Create configurations
    try:
        json_paths = create_variant_configs(variants, args.output)
        print(f"\nCreated {len(json_paths)} JSON configuration files")
    except Exception as e:
        print(f"Error creating configurations: {e}", file=sys.stderr)
        sys.exit(1)

    # Show first few paths
    print("\nConfiguration files:")
    for path in json_paths[:3]:
        print(f"  {path}")
    if len(json_paths) > 3:
        print(f"  ... and {len(json_paths) - 3} more")

    if args.json_only:
        print("\nJSON configurations created. Use --json-only=false to run predictions.")
        return

    # Instructions for running batch predictions
    print("\nTo run batch AlphaFold3 predictions:")
    print("1. Activate the conda environment:")
    print("   mamba activate ./env")
    print()
    print("2. Run batch predictions using the MCP tool:")
    print(f"   python src/tools/af3_predict_structure.py \\")
    print(f"     --batch_dir={args.output} \\")
    print(f"     --model_dir=repo/alphafold3/model \\")
    print(f"     --db_dir=repo/alphafold3/alphafold3_db")
    print()
    print("   OR run individual predictions:")
    for i, path in enumerate(json_paths[:2]):
        variant_dir = Path(path).parent
        print(f"   # Variant {i+1}")
        print(f"   python src/tools/af3_predict_structure.py \\")
        print(f"     --json_path={path} \\")
        print(f"     --model_dir=repo/alphafold3/model \\")
        print(f"     --db_dir=repo/alphafold3/alphafold3_db \\")
        print(f"     --output_dir={variant_dir}")
        print()

    print("3. Results will be saved in each variant's subdirectory:")
    print(f"   {args.output}/seq_0/{variants[0]['name']}_model.cif")
    print(f"   {args.output}/seq_1/{variants[1]['name']}_model.cif")
    print("   ...")
    print()
    print("Note: Batch predictions can take significant time. Consider running")
    print("      variants in parallel if you have multiple GPUs available.")


if __name__ == "__main__":
    main()