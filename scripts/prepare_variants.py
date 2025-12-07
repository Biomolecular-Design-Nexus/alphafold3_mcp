#!/usr/bin/env python3
"""Python interface for preparing AlphaFold3 variant configurations.

This module provides Python functions to prepare AlphaFold3 input configurations
for protein variants. It uses the wild-type MSA and templates to create input
files for variants, enabling efficient batch processing.

Example usage:
    from prepare_variants import (
        prepare_variant_configs,
        parse_fasta,
        load_wt_data,
        create_variant_input,
    )

    # Prepare all variants from a FASTA file
    prepare_variant_configs(
        variants_fasta="/path/to/variants.fasta",
        wt_data_json="/path/to/wt_data.json",
        output_dir="/path/to/output",
    )

    # Prepare a single variant programmatically
    wt_data = load_wt_data("/path/to/wt_data.json")
    input_json = create_variant_input(
        wt_data=wt_data,
        variant_name="A50V",
        variant_sequence="MKVL...",
    )
"""

import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from loguru import logger


# Configure loguru for variant preparation
def _configure_logger():
    """Configure loguru logger for variant preparation output."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>Variants</cyan> | {message}",
        level="INFO",
        colorize=True,
    )


# Initialize logger configuration
_configure_logger()


def _get_af3_path() -> Path:
    """Get the AlphaFold3 repository path."""
    script_dir = Path(__file__).parent.absolute()
    af3_path = script_dir.parent / "repo" / "alphafold3"

    if not af3_path.exists():
        raise FileNotFoundError(
            f"AlphaFold3 repository not found at {af3_path}. "
            "Please set the AF3_PATH environment variable."
        )
    return af3_path


@dataclass
class VariantInfo:
    """Information about a protein variant."""

    name: str
    sequence: str
    mutations: Optional[str] = None  # e.g., "A50V,K100R"


@dataclass
class WildTypeData:
    """Wild-type data containing MSA and templates."""

    name: str
    sequence: str
    unpaired_msa: str
    paired_msa: str
    templates: list
    raw_data: dict

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "WildTypeData":
        """Load wild-type data from an AF3 output JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract protein entity
        protein = None
        sequence = ""
        for seq_entry in data.get("sequences", []):
            if "protein" in seq_entry:
                protein = seq_entry["protein"]
                sequence = protein.get("sequence", "")
                break

        if protein is None:
            raise ValueError("No protein entity found in wild-type data JSON")

        return cls(
            name=data.get("name", "unknown"),
            sequence=sequence,
            unpaired_msa=protein.get("unpairedMsa", ""),
            paired_msa=protein.get("pairedMsa", ""),
            templates=protein.get("templates", []),
            raw_data=data,
        )


def parse_fasta(fasta_path: Union[str, Path]) -> list[VariantInfo]:
    """Parse a FASTA file and return list of VariantInfo objects.

    Args:
        fasta_path: Path to FASTA file containing variant sequences.

    Returns:
        List of VariantInfo objects with name and sequence.

    Example:
        variants = parse_fasta("variants.fasta")
        for v in variants:
            print(f"{v.name}: {len(v.sequence)} aa")
    """
    sequences = []
    current_name = None
    current_seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # Save previous sequence
                if current_name is not None:
                    sequences.append(
                        VariantInfo(
                            name=current_name,
                            sequence="".join(current_seq_parts),
                        )
                    )
                # Start new sequence
                current_name = line[1:].split()[0]  # Get ID before first space
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        # Don't forget the last sequence
        if current_name is not None:
            sequences.append(
                VariantInfo(
                    name=current_name,
                    sequence="".join(current_seq_parts),
                )
            )

    return sequences


def load_wt_data(json_path: Union[str, Path]) -> WildTypeData:
    """Load wild-type data from an AF3 output JSON file.

    This JSON file should contain processed MSA and templates from a previous
    AlphaFold3 run. It can be found at:
    <output_dir>/<job_name>_data.json

    Args:
        json_path: Path to the wild-type data JSON file.

    Returns:
        WildTypeData object containing MSA, templates, and sequence.

    Example:
        wt_data = load_wt_data("wt_prediction/wt_protein_data.json")
        print(f"Wild-type: {wt_data.name}")
        print(f"Sequence length: {len(wt_data.sequence)}")
        print(f"Templates: {len(wt_data.templates)}")
    """
    return WildTypeData.from_json(json_path)


def create_variant_msa(
    wt_msa: str,
    variant_name: str,
    variant_sequence: str,
) -> str:
    """Create a variant MSA by replacing the query sequence.

    The first entry in the MSA (query sequence) is replaced with the variant
    sequence while preserving all other MSA entries.

    Args:
        wt_msa: The unpaired MSA string from wild-type data.
        variant_name: Name for the variant (used in header).
        variant_sequence: The variant protein sequence.

    Returns:
        MSA content as a string with the variant sequence as query.
    """
    lines = wt_msa.split("\n")
    result_lines = []
    first_entry = True
    skip_sequence = False

    for line in lines:
        if line.startswith(">"):
            if first_entry:
                # Replace first header with variant name
                result_lines.append(f">{variant_name}")
                skip_sequence = True
                first_entry = False
            else:
                result_lines.append(line)
                skip_sequence = False
        else:
            if skip_sequence:
                # Replace first sequence with variant
                result_lines.append(variant_sequence)
                skip_sequence = False
            else:
                result_lines.append(line)

    return "\n".join(result_lines)


def create_variant_input(
    wt_data: WildTypeData,
    variant_name: str,
    variant_sequence: str,
    ligand_smiles: Optional[str] = None,
    ligand_id: str = "B",
    model_seeds: Optional[list[int]] = None,
) -> dict:
    """Create an AlphaFold3 input JSON for a variant.

    This creates a new input JSON by:
    1. Copying the wild-type data structure
    2. Replacing the protein sequence with the variant
    3. Modifying the MSA to use the variant as the query sequence
    4. Keeping the templates from wild-type (valid for point mutations)
    5. Optionally adding a ligand

    Args:
        wt_data: Wild-type data containing MSA and templates.
        variant_name: Name for the variant prediction job.
        variant_sequence: The variant protein sequence.
        ligand_smiles: Optional SMILES string for a ligand to add.
        ligand_id: Chain ID for the ligand (default: 'B').
        model_seeds: Random seeds for prediction (overrides wt_data if provided).

    Returns:
        Dictionary ready to be saved as input.json.

    Raises:
        ValueError: If variant sequence length doesn't match wild-type.

    Example:
        wt_data = load_wt_data("wt_data.json")
        input_json = create_variant_input(
            wt_data=wt_data,
            variant_name="A50V",
            variant_sequence=wt_data.sequence[:49] + "V" + wt_data.sequence[50:],
        )
    """
    # Validate sequence lengths match
    if len(variant_sequence) != len(wt_data.sequence):
        raise ValueError(
            f"Variant sequence length ({len(variant_sequence)}) doesn't match "
            f"wild-type sequence length ({len(wt_data.sequence)}). "
            f"Only point mutations are supported."
        )

    # Deep copy to avoid modifying original
    result = copy.deepcopy(wt_data.raw_data)

    # Update name
    result["name"] = variant_name

    # Update model seeds if provided
    if model_seeds is not None:
        result["modelSeeds"] = model_seeds

    # Find and update the protein entity
    protein_found = False
    for seq_entry in result.get("sequences", []):
        if "protein" in seq_entry:
            protein = seq_entry["protein"]
            protein_found = True

            # Update sequence
            protein["sequence"] = variant_sequence

            # Update unpaired MSA if present
            if "unpairedMsa" in protein and protein["unpairedMsa"]:
                protein["unpairedMsa"] = create_variant_msa(
                    protein["unpairedMsa"],
                    variant_name,
                    variant_sequence,
                )

            # Templates are kept as-is since they're structural templates
            # and remain valid for point mutations
            break

    if not protein_found:
        raise ValueError("No protein entity found in wild-type data")

    # Add ligand if specified and not already present
    if ligand_smiles:
        has_ligand = any("ligand" in seq for seq in result.get("sequences", []))
        if not has_ligand:
            result["sequences"].append(
                {
                    "ligand": {
                        "id": ligand_id,
                        "smiles": ligand_smiles,
                    }
                }
            )

    return result


def prepare_variant_configs(
    variants_fasta: Union[str, Path],
    wt_data_json: Union[str, Path],
    output_dir: Union[str, Path],
    ligand_smiles: Optional[str] = None,
    ligand_id: str = "B",
    model_seeds: Optional[list[int]] = None,
    skip_length_mismatch: bool = True,
) -> dict:
    """Prepare AlphaFold3 input configs for multiple variants.

    This function reads variant sequences from a FASTA file and creates
    input.json files for each variant using the MSA and templates from
    a wild-type prediction.

    Args:
        variants_fasta: Path to FASTA file containing variant sequences.
        wt_data_json: Path to wild-type AF3 data JSON file.
        output_dir: Output directory for variant configs.
        ligand_smiles: Optional SMILES string for a ligand to include.
        ligand_id: Chain ID for the ligand (default: 'B').
        model_seeds: Random seeds for prediction (default: use seeds from wt_data).
        skip_length_mismatch: Skip variants with different lengths (default: True).

    Returns:
        Dictionary with 'created', 'skipped', and 'variant_dirs' keys.

    Example:
        result = prepare_variant_configs(
            variants_fasta="variants.fasta",
            wt_data_json="wt_data.json",
            output_dir="variants",
            ligand_smiles="CCO",  # Optional ligand
        )
        print(f"Created {result['created']} configs")

    Directory structure created:
        output_dir/
        ├── variant1/
        │   └── input.json
        ├── variant2/
        │   └── input.json
        └── ...
    """
    # Load wild-type data
    logger.info(f"Loading wild-type data from {wt_data_json}")
    wt_data = load_wt_data(wt_data_json)
    logger.info(f"Wild-type: {wt_data.name} ({len(wt_data.sequence)} aa)")
    logger.debug(f"  Templates: {len(wt_data.templates)}")

    if wt_data.unpaired_msa:
        msa_count = wt_data.unpaired_msa.count(">")
        logger.debug(f"  Unpaired MSA: {msa_count} sequences")

    # Load variant sequences
    logger.info(f"Loading variants from {variants_fasta}")
    variants = parse_fasta(variants_fasta)
    if not variants:
        raise ValueError(f"No sequences found in: {variants_fasta}")
    logger.info(f"Loaded {len(variants)} variant sequences")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process variants
    created = 0
    skipped = 0
    variant_dirs = []

    for variant in variants:
        # Check sequence length matches
        if len(variant.sequence) != len(wt_data.sequence):
            if skip_length_mismatch:
                logger.warning(
                    f"Skipping {variant.name} - length mismatch "
                    f"({len(variant.sequence)} vs {len(wt_data.sequence)} aa)"
                )
                skipped += 1
                continue
            else:
                raise ValueError(
                    f"Variant {variant.name} has different length "
                    f"({len(variant.sequence)} vs {len(wt_data.sequence)} aa)"
                )

        # Create variant directory
        variant_dir = output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Create input JSON
        input_json = create_variant_input(
            wt_data=wt_data,
            variant_name=variant.name,
            variant_sequence=variant.sequence,
            ligand_smiles=ligand_smiles,
            ligand_id=ligand_id,
            model_seeds=model_seeds,
        )

        # Save input JSON
        json_path = variant_dir / "input.json"
        with open(json_path, "w") as f:
            json.dump(input_json, f, indent=2)

        logger.debug(f"Created config for {variant.name}")
        created += 1
        variant_dirs.append(str(variant_dir))

    logger.success(f"Created {created} variant configs in {output_dir}")
    if skipped > 0:
        logger.warning(f"Skipped {skipped} variants due to length mismatch")

    return {
        "created": created,
        "skipped": skipped,
        "variant_dirs": variant_dirs,
    }


def prepare_variant_configs_cli(
    variants_fasta: Union[str, Path],
    wt_data_json: Union[str, Path],
    output_dir: Union[str, Path],
    ligand_smiles: Optional[str] = None,
    ligand_id: str = "B",
    model_seeds: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run the prepare_variant_af3_configs.py script via CLI.

    This is an alternative to prepare_variant_configs() that runs the
    original script as a subprocess.

    Args:
        variants_fasta: Path to FASTA file containing variant sequences.
        wt_data_json: Path to wild-type AF3 data JSON file.
        output_dir: Output directory for variant configs.
        ligand_smiles: Optional SMILES string for a ligand.
        ligand_id: Chain ID for the ligand.
        model_seeds: Comma-separated model seeds string.

    Returns:
        subprocess.CompletedProcess with return code and output.
    """
    af3_path = _get_af3_path()

    cmd = [
        sys.executable,
        str(af3_path / "prepare_variant_af3_configs.py"),
        f"--variants_fasta={variants_fasta}",
        f"--wt_data_json={wt_data_json}",
        f"--output_dir={output_dir}",
    ]

    if ligand_smiles:
        cmd.append(f"--ligand_smiles={ligand_smiles}")
    if ligand_id:
        cmd.append(f"--ligand_id={ligand_id}")
    if model_seeds:
        cmd.append(f"--model_seeds={model_seeds}")

    logger.info(f"Running prepare_variant_af3_configs.py")
    return subprocess.run(cmd, capture_output=True, text=True, cwd=af3_path)


# Convenience function to prepare and run
def prepare_and_run_variants(
    variants_fasta: Union[str, Path],
    wt_data_json: Union[str, Path],
    output_dir: Union[str, Path],
    device: int = 0,
    ligand_smiles: Optional[str] = None,
    skip_existing: bool = True,
) -> dict:
    """Prepare variant configs and run batch prediction.

    This is a convenience function that combines prepare_variant_configs()
    and run_batch() into a single call.

    Args:
        variants_fasta: Path to FASTA file containing variant sequences.
        wt_data_json: Path to wild-type AF3 data JSON file.
        output_dir: Output directory for variant configs and predictions.
        device: GPU device number (default: 0).
        ligand_smiles: Optional SMILES string for a ligand.
        skip_existing: Skip already completed predictions.

    Returns:
        Dictionary with 'prepare_result' and 'prediction_result' keys.

    Example:
        result = prepare_and_run_variants(
            variants_fasta="variants.fasta",
            wt_data_json="wt_data.json",
            output_dir="variants",
            device=0,
        )
    """
    # Import here to avoid circular imports
    from alphafold3_runner import run_batch

    logger.info("Starting variant preparation and prediction workflow")

    # Prepare configs
    logger.info("Step 1: Preparing variant configs")
    prepare_result = prepare_variant_configs(
        variants_fasta=variants_fasta,
        wt_data_json=wt_data_json,
        output_dir=output_dir,
        ligand_smiles=ligand_smiles,
    )

    # Run batch prediction
    logger.info("Step 2: Running batch prediction")
    prediction_result = run_batch(
        input_dir=output_dir,
        device=device,
        skip_existing=skip_existing,
        run_template_search=False,  # Templates already in input.json
    )

    logger.success("Variant workflow complete!")
    return {
        "prepare_result": prepare_result,
        "prediction_result": prediction_result,
    }


def set_log_level(level: str = "INFO") -> None:
    """Set the logging level for variant preparation output.

    Args:
        level: Log level - DEBUG, INFO, WARNING, ERROR, or SUCCESS

    Example:
        set_log_level("DEBUG")  # Show all logs including debug
        set_log_level("WARNING")  # Only show warnings and errors
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>Variants</cyan> | {message}",
        level=level.upper(),
        colorize=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare AlphaFold3 variant configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare variant configs
  python prepare_variants.py \\
    --variants-fasta variants.fasta \\
    --wt-data-json wt_data.json \\
    --output-dir variants

  # With ligand
  python prepare_variants.py \\
    --variants-fasta variants.fasta \\
    --wt-data-json wt_data.json \\
    --output-dir variants \\
    --ligand-smiles "CCO"

  # With debug logging
  python prepare_variants.py \\
    --variants-fasta variants.fasta \\
    --wt-data-json wt_data.json \\
    --output-dir variants \\
    --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--variants-fasta",
        required=True,
        help="Path to FASTA file containing variant sequences",
    )
    parser.add_argument(
        "--wt-data-json",
        required=True,
        help="Path to wild-type AF3 data JSON file",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for variant configs",
    )
    parser.add_argument(
        "--ligand-smiles",
        default=None,
        help="Optional SMILES string for a ligand",
    )
    parser.add_argument(
        "--ligand-id",
        default="B",
        help="Chain ID for the ligand (default: B)",
    )
    parser.add_argument(
        "--model-seeds",
        default=None,
        help="Comma-separated list of model seeds",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    set_log_level(args.log_level)

    # Parse model seeds
    model_seeds = None
    if args.model_seeds:
        model_seeds = [int(s.strip()) for s in args.model_seeds.split(",")]

    try:
        result = prepare_variant_configs(
            variants_fasta=args.variants_fasta,
            wt_data_json=args.wt_data_json,
            output_dir=args.output_dir,
            ligand_smiles=args.ligand_smiles,
            ligand_id=args.ligand_id,
            model_seeds=model_seeds,
        )
        logger.success(f"Complete: {result['created']} configs created")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
