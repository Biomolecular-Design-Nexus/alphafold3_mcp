#!/usr/bin/env python3
"""
Script: batch_variants_prediction.py
Description: Create AlphaFold3 configurations for batch protein variant prediction

Original Use Case: examples/use_case_4_batch_variants.py
Dependencies Removed: None (only uses Python standard library)

Usage:
    python scripts/clean/batch_variants_prediction.py --fasta variants.fasta --output-dir variants_output
    python scripts/clean/batch_variants_prediction.py --variants mutations.txt --template template.fasta --output-dir results

Example:
    python scripts/clean/batch_variants_prediction.py --fasta examples/data/protein_variants.fasta --output-dir batch_results
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_seeds": [1],
    "dialect": "alphafold3",
    "version": 1,
    "max_variants": None,  # No limit by default
    "variant_dir_prefix": "seq"
}

# ==============================================================================
# Inlined Utility Functions (simplified from use case)
# ==============================================================================
def read_fasta_single(file_path: Path) -> tuple[str, str]:
    """Read a single-sequence FASTA file."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or not lines[0].startswith('>'):
        raise ValueError("Invalid FASTA format")

    name = lines[0][1:]  # Remove '>' prefix
    sequence = ''.join(lines[1:])  # Join all sequence lines
    return name, sequence


def parse_fasta_variants(fasta_path: Path) -> List[Dict[str, str]]:
    """Parse a FASTA file containing multiple protein variants. Inlined from use case."""
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


def apply_mutations(template_seq: str, mutation_str: str) -> tuple[str, List[str]]:
    """Apply mutations to a template sequence. Inlined from use case."""
    sequence = list(template_seq)
    warnings = []

    for mutation in mutation_str.split():
        mutation = mutation.strip()
        if not mutation or len(mutation) < 3:
            continue

        # Parse mutation (e.g., A123V)
        orig_aa = mutation[0]
        new_aa = mutation[-1]
        pos_str = mutation[1:-1]

        try:
            pos = int(pos_str) - 1  # Convert to 0-based
        except ValueError:
            warnings.append(f"Invalid mutation format: {mutation}")
            continue

        if 0 <= pos < len(sequence):
            if sequence[pos].upper() != orig_aa.upper():
                warnings.append(f"Position {pos+1} is {sequence[pos]}, not {orig_aa}")
            sequence[pos] = new_aa.upper()
        else:
            warnings.append(f"Position {pos+1} out of range for mutation {mutation}")

    return ''.join(sequence), warnings


def parse_variant_list(variant_file: Path, template_fasta: Path) -> tuple[List[Dict[str, str]], List[str]]:
    """Parse a text file with variant definitions and apply to template. Inlined from use case."""
    # Read template sequence
    template_name, template_seq = read_fasta_single(template_fasta)

    # Read variant definitions
    variants = []
    all_warnings = []

    with open(variant_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Apply mutations to template
            variant_seq, warnings = apply_mutations(template_seq, line)
            variant_name = f"{template_name}_var_{i+1}_{line.replace(' ', '_')}"

            variants.append({
                'name': variant_name,
                'sequence': variant_seq
            })

            all_warnings.extend([f"Variant {i+1} ({line}): {w}" for w in warnings])

    return variants, all_warnings


def validate_protein_sequence(sequence: str) -> bool:
    """Validate that the sequence contains only valid amino acids. Inlined from use case."""
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa.upper() in valid_aa for aa in sequence)


def create_variant_configs(variants: List[Dict[str, str]], output_dir: Path, config: Dict[str, Any]) -> List[Path]:
    """Create AlphaFold3 JSON configurations for all variants. Inlined from use case."""
    json_paths = []

    for i, variant in enumerate(variants):
        # Create variant subdirectory
        variant_dir = output_dir / f"{config['variant_dir_prefix']}_{i}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Create configuration
        af3_config = {
            "name": variant['name'],
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": variant['sequence']
                    }
                }
            ],
            "modelSeeds": config["model_seeds"],
            "dialect": config["dialect"],
            "version": config["version"]
        }

        # Write JSON file
        json_path = variant_dir / "input.json"
        with open(json_path, 'w') as f:
            json.dump(af3_config, f, indent=2)

        json_paths.append(json_path)

    return json_paths


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_batch_variants_prediction(
    fasta_file: Optional[Union[str, Path]] = None,
    variants_file: Optional[Union[str, Path]] = None,
    template_file: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "batch_output",
    max_variants: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for batch protein variant prediction configuration.

    Args:
        fasta_file: Path to FASTA file with multiple variants (mutually exclusive with variants_file)
        variants_file: Path to variant definitions file (requires template_file)
        template_file: Path to template FASTA file (required with variants_file)
        output_dir: Directory to save variant configurations
        max_variants: Maximum number of variants to process (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - variant_count: Number of variants processed
            - config_files: List of paths to generated JSON files
            - output_dir: Output directory path
            - warnings: List of any warnings encountered
            - metadata: Execution metadata

    Example:
        >>> result = run_batch_variants_prediction(
        ...     fasta_file="variants.fasta",
        ...     output_dir="batch_results"
        ... )
        >>> print(f"Generated {result['variant_count']} configs")
    """
    # Setup
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}
    if max_variants is not None:
        config['max_variants'] = max_variants

    output_dir = Path(output_dir)

    # Validate input - exactly one input source
    if not ((fasta_file is None) ^ (variants_file is None)):
        raise ValueError("Exactly one of 'fasta_file' or 'variants_file' must be provided")

    if variants_file and not template_file:
        raise ValueError("'template_file' is required when using 'variants_file'")

    # Parse input variants
    warnings = []

    if fasta_file:
        fasta_file = Path(fasta_file)
        if not fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

        variants = parse_fasta_variants(fasta_file)
        input_desc = f"FASTA file: {fasta_file}"

    else:  # variants_file
        variants_file = Path(variants_file)
        template_file = Path(template_file)

        if not variants_file.exists():
            raise FileNotFoundError(f"Variants file not found: {variants_file}")
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")

        variants, warnings = parse_variant_list(variants_file, template_file)
        input_desc = f"Variant list: {variants_file}, Template: {template_file}"

    if not variants:
        raise ValueError("No variants found in input")

    # Apply max variants limit
    if config['max_variants'] and len(variants) > config['max_variants']:
        original_count = len(variants)
        variants = variants[:config['max_variants']]
        warnings.append(f"Limited to first {config['max_variants']} variants (out of {original_count})")

    # Validate all sequences
    invalid_variants = []
    for i, variant in enumerate(variants):
        if not validate_protein_sequence(variant['sequence']):
            invalid_variants.append(f"{i+1}: {variant['name']}")

    if invalid_variants:
        raise ValueError(f"Invalid protein sequences found: {', '.join(invalid_variants)}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create configurations
    json_paths = create_variant_configs(variants, output_dir, config)

    return {
        "variant_count": len(variants),
        "config_files": [str(p) for p in json_paths],
        "output_dir": str(output_dir),
        "warnings": warnings,
        "metadata": {
            "input_description": input_desc,
            "variants_summary": [
                {"name": v["name"], "sequence_length": len(v["sequence"])}
                for v in variants[:5]  # First 5 for summary
            ],
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
    input_group.add_argument('--fasta', help='FASTA file containing multiple protein variants')
    input_group.add_argument('--variants', help='Text file with variant definitions (requires --template)')

    # Template for variant list mode
    parser.add_argument('--template', help='Template FASTA file for variant list mode')

    # Output options
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for all variants')
    parser.add_argument('--max-variants', type=int, help='Maximum number of variants to process')

    # Configuration options
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1],
                       help='Model seeds (default: [1])')

    args = parser.parse_args()

    # Validate input combination
    if args.variants and not args.template:
        parser.error("--template is required when using --variants")

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
    result = run_batch_variants_prediction(
        fasta_file=args.fasta,
        variants_file=args.variants,
        template_file=args.template,
        output_dir=getattr(args, 'output_dir'),
        max_variants=getattr(args, 'max_variants'),
        config=config
    )

    # Print results
    metadata = result['metadata']
    print(f"✅ Batch variant configurations created")
    print(f"   Input: {metadata['input_description']}")
    print(f"   Variants processed: {result['variant_count']}")
    print(f"   Output directory: {result['output_dir']}")

    # Show warnings
    if result['warnings']:
        print(f"   Warnings: {len(result['warnings'])}")
        for warning in result['warnings'][:3]:  # Show first 3 warnings
            print(f"     • {warning}")
        if len(result['warnings']) > 3:
            print(f"     • ... and {len(result['warnings']) - 3} more warnings")

    # Show variant summary
    print(f"   Config files: {len(result['config_files'])}")
    for i, summary in enumerate(metadata['variants_summary']):
        print(f"     {i+1}. {summary['name']} ({summary['sequence_length']} aa)")

    if result['variant_count'] > len(metadata['variants_summary']):
        remaining = result['variant_count'] - len(metadata['variants_summary'])
        print(f"     ... and {remaining} more variants")

    return result

if __name__ == '__main__':
    main()