"""
AlphaFold3 variant preparation tools.

This MCP Server provides tools for preparing AlphaFold3 variant configurations:
1. af3_prepare_variants: Prepare input configs for multiple protein variants
2. af3_prepare_and_predict_variants: Combined preparation and prediction workflow

The tools use wild-type MSA and templates to create variant input files,
enabling efficient batch processing of protein variants.
"""

import copy
import json
from pathlib import Path
from typing import Annotated, Optional

from fastmcp import FastMCP
from loguru import logger

# MCP server instance
af3_variants_mcp = FastMCP(name="af3_prepare_variants")


def _parse_fasta(fasta_path: Path) -> list[dict]:
    """Parse a FASTA file and return list of variant info dicts."""
    logger.debug(f"Parsing FASTA file: {fasta_path}")
    sequences = []
    current_name = None
    current_seq_parts = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    sequences.append({
                        "name": current_name,
                        "sequence": "".join(current_seq_parts),
                    })
                current_name = line[1:].split()[0]
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

        if current_name is not None:
            sequences.append({
                "name": current_name,
                "sequence": "".join(current_seq_parts),
            })

    logger.debug(f"Parsed {len(sequences)} sequences from FASTA")
    return sequences


def _load_wt_data(json_path: Path) -> dict:
    """Load wild-type data from an AF3 output JSON file."""
    logger.debug(f"Loading wild-type data from: {json_path}")
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
        logger.error("No protein entity found in wild-type data JSON")
        raise ValueError("No protein entity found in wild-type data JSON")

    logger.debug(f"Loaded wild-type sequence of length {len(sequence)}")
    return {
        "name": data.get("name", "unknown"),
        "sequence": sequence,
        "unpaired_msa": protein.get("unpairedMsa", ""),
        "paired_msa": protein.get("pairedMsa", ""),
        "templates": protein.get("templates", []),
        "raw_data": data,
    }


def _create_variant_msa(wt_msa: str, variant_name: str, variant_sequence: str) -> str:
    """Create a variant MSA by replacing the query sequence."""
    lines = wt_msa.split("\n")
    result_lines = []
    first_entry = True
    skip_sequence = False

    for line in lines:
        if line.startswith(">"):
            if first_entry:
                result_lines.append(f">{variant_name}")
                skip_sequence = True
                first_entry = False
            else:
                result_lines.append(line)
                skip_sequence = False
        else:
            if skip_sequence:
                result_lines.append(variant_sequence)
                skip_sequence = False
            else:
                result_lines.append(line)

    return "\n".join(result_lines)


def _create_variant_input(
    wt_data: dict,
    variant_name: str,
    variant_sequence: str,
    ligand_smiles: Optional[str] = None,
    ligand_id: str = "B",
    model_seeds: Optional[list[int]] = None,
) -> dict:
    """Create an AlphaFold3 input JSON for a variant."""
    # Validate sequence lengths match
    if len(variant_sequence) != len(wt_data["sequence"]):
        raise ValueError(
            f"Variant sequence length ({len(variant_sequence)}) doesn't match "
            f"wild-type sequence length ({len(wt_data['sequence'])}). "
            f"Only point mutations are supported."
        )

    # Deep copy to avoid modifying original
    result = copy.deepcopy(wt_data["raw_data"])

    # Update name
    result["name"] = variant_name

    # Update model seeds if provided
    if model_seeds is not None:
        result["modelSeeds"] = model_seeds

    # Find and update the protein entity
    for seq_entry in result.get("sequences", []):
        if "protein" in seq_entry:
            protein = seq_entry["protein"]

            # Update sequence
            protein["sequence"] = variant_sequence

            # Update unpaired MSA if present
            if "unpairedMsa" in protein and protein["unpairedMsa"]:
                protein["unpairedMsa"] = _create_variant_msa(
                    protein["unpairedMsa"],
                    variant_name,
                    variant_sequence,
                )
            break

    # Add ligand if specified
    if ligand_smiles:
        has_ligand = any("ligand" in seq for seq in result.get("sequences", []))
        if not has_ligand:
            result["sequences"].append({
                "ligand": {
                    "id": ligand_id,
                    "smiles": ligand_smiles,
                }
            })

    return result


def _af3_prepare_variants_impl(
    variants_fasta: str,
    wt_data_json: str,
    output_dir: str,
    ligand_smiles: Optional[str] = None,
    ligand_id: str = "B",
    model_seeds: Optional[str] = None,
    skip_length_mismatch: bool = True,
) -> dict:
    """Internal implementation of af3_prepare_variants."""
    logger.info(f"af3_prepare_variants called with variants_fasta={variants_fasta}, wt_data_json={wt_data_json}")

    try:
        variants_fasta = Path(variants_fasta)
        wt_data_json = Path(wt_data_json)
        output_dir = Path(output_dir)

        if not variants_fasta.exists():
            logger.error(f"Variants FASTA not found: {variants_fasta}")
            raise FileNotFoundError(f"Variants FASTA not found: {variants_fasta}")
        if not wt_data_json.exists():
            logger.error(f"Wild-type data JSON not found: {wt_data_json}")
            raise FileNotFoundError(f"Wild-type data JSON not found: {wt_data_json}")

        # Load wild-type data
        wt_data = _load_wt_data(wt_data_json)
        logger.info(f"Loaded wild-type data: {wt_data['name']}, sequence length: {len(wt_data['sequence'])}")

        # Parse model seeds
        seeds = None
        if model_seeds:
            seeds = [int(s.strip()) for s in model_seeds.split(",")]

        # Load variants
        variants = _parse_fasta(variants_fasta)

        if not variants:
            logger.error(f"No sequences found in: {variants_fasta}")
            raise ValueError(f"No sequences found in: {variants_fasta}")

        logger.info(f"Found {len(variants)} variants to process")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process variants
        created = 0
        skipped = 0
        skipped_names = []
        variant_dirs = []

        for variant in variants:
            # Check sequence length matches
            if len(variant["sequence"]) != len(wt_data["sequence"]):
                if skip_length_mismatch:
                    skipped += 1
                    skipped_names.append(variant["name"])
                    logger.warning(f"Skipping variant '{variant['name']}' due to length mismatch "
                                   f"({len(variant['sequence'])} vs {len(wt_data['sequence'])} aa)")
                    continue
                else:
                    logger.error(f"Variant {variant['name']} has different length")
                    raise ValueError(
                        f"Variant {variant['name']} has different length "
                        f"({len(variant['sequence'])} vs {len(wt_data['sequence'])} aa)"
                    )

            # Create variant directory
            variant_dir = output_dir / variant["name"]
            variant_dir.mkdir(parents=True, exist_ok=True)

            # Create input JSON
            input_json = _create_variant_input(
                wt_data=wt_data,
                variant_name=variant["name"],
                variant_sequence=variant["sequence"],
                ligand_smiles=ligand_smiles,
                ligand_id=ligand_id,
                model_seeds=seeds,
            )

            # Save input JSON
            json_path = variant_dir / "input.json"
            with open(json_path, "w") as f:
                json.dump(input_json, f, indent=2)

            created += 1
            variant_dirs.append(str(variant_dir))

        logger.info(f"Variant preparation completed. Created: {created}, Skipped: {skipped}")

        return {
            "status": "success",
            "variants_fasta": str(variants_fasta),
            "wt_data_json": str(wt_data_json),
            "output_dir": str(output_dir),
            "total_variants": len(variants),
            "created": created,
            "skipped": skipped,
            "skipped_names": skipped_names[:10] if len(skipped_names) > 10 else skipped_names,
            "variant_dirs": variant_dirs[:10] if len(variant_dirs) > 10 else variant_dirs,
            "wt_name": wt_data["name"],
            "wt_sequence_length": len(wt_data["sequence"]),
            "has_ligand": ligand_smiles is not None,
        }

    except Exception as e:
        logger.exception(f"Exception during variant preparation: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "variants_fasta": str(variants_fasta),
            "wt_data_json": str(wt_data_json),
        }


@af3_variants_mcp.tool
def af3_prepare_variants(
    variants_fasta: Annotated[str, "Path to FASTA file containing variant sequences"],
    wt_data_json: Annotated[str, "Path to wild-type AF3 data JSON file"],
    output_dir: Annotated[str, "Output directory for variant configs"],
    ligand_smiles: Annotated[Optional[str], "Optional SMILES string for a ligand to include"] = None,
    ligand_id: Annotated[str, "Chain ID for the ligand"] = "B",
    model_seeds: Annotated[Optional[str], "Comma-separated model seeds (e.g., '1,2,3'). If None, uses wild-type seeds"] = None,
    skip_length_mismatch: Annotated[bool, "Skip variants with different lengths"] = True,
) -> dict:
    """
    Prepare AlphaFold3 input configs for multiple protein variants.

    This tool reads variant sequences from a FASTA file and creates input.json
    files for each variant using the MSA and templates from a wild-type prediction.

    Output directory structure:
        output_dir/
        ├── variant1/
        │   └── input.json
        ├── variant2/
        │   └── input.json
        └── ...

    Note: Only point mutations (same length) are supported by default.
    Set skip_length_mismatch=False to raise errors for length mismatches.

    Input: FASTA with variants, wild-type data JSON, output directory
    Output: Dictionary with counts of created/skipped configs
    """
    return _af3_prepare_variants_impl(
        variants_fasta=variants_fasta,
        wt_data_json=wt_data_json,
        output_dir=output_dir,
        ligand_smiles=ligand_smiles,
        ligand_id=ligand_id,
        model_seeds=model_seeds,
        skip_length_mismatch=skip_length_mismatch,
    )


@af3_variants_mcp.tool
def af3_prepare_and_predict_variants(
    variants_fasta: Annotated[str, "Path to FASTA file containing variant sequences"],
    wt_data_json: Annotated[str, "Path to wild-type AF3 data JSON file"],
    output_dir: Annotated[str, "Output directory for variant configs and predictions"],
    device: Annotated[int, "GPU device number"] = 0,
    ligand_smiles: Annotated[Optional[str], "Optional SMILES string for a ligand"] = None,
    skip_existing: Annotated[bool, "Skip already completed predictions"] = True,
    flash_attention_implementation: Annotated[str, "Flash attention implementation"] = "triton",
) -> dict:
    """
    Prepare variant configs and run batch prediction (convenience workflow).

    This tool combines variant preparation and batch prediction:
    1. Prepares variant input.json files from FASTA
    2. Runs batch AlphaFold3 predictions

    This is a convenience function for the common workflow of predicting
    structures for multiple protein variants.

    Input: Variants FASTA, wild-type data JSON, output directory
    Output: Dictionary with preparation and prediction results
    """
    logger.info(f"af3_prepare_and_predict_variants called with variants_fasta={variants_fasta}")

    try:
        # Import the internal implementation function (not the decorated tool)
        from .af3_predict_structure import _af3_predict_batch_impl

        # Step 1: Prepare variants (call internal impl directly)
        logger.info("Step 1/2: Preparing variant configs")
        prepare_result = _af3_prepare_variants_impl(
            variants_fasta=variants_fasta,
            wt_data_json=wt_data_json,
            output_dir=output_dir,
            ligand_smiles=ligand_smiles,
        )

        if prepare_result["status"] != "success":
            error_msg = f"Failed to prepare variants: {prepare_result.get('error_message', 'Unknown error')}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error_message": error_msg,
                "prepare_result": prepare_result,
            }

        # Step 2: Run batch prediction (call internal impl directly)
        logger.info("Step 2/2: Running batch predictions")
        predict_result = _af3_predict_batch_impl(
            input_dir=output_dir,
            device=device,
            skip_existing=skip_existing,
            flash_attention_implementation=flash_attention_implementation,
        )

        if predict_result["status"] == "success":
            logger.info(f"Combined workflow completed successfully. "
                        f"Variants prepared: {prepare_result['created']}, "
                        f"Predictions completed: {predict_result.get('completed_count', 0)}")
        else:
            logger.warning(f"Combined workflow completed with errors. "
                           f"Variants prepared: {prepare_result['created']}, "
                           f"Predictions completed: {predict_result.get('completed_count', 0)}")

        return {
            "status": "success" if predict_result["status"] == "success" else "partial",
            "prepare_result": prepare_result,
            "predict_result": predict_result,
            "variants_prepared": prepare_result["created"],
            "predictions_completed": predict_result.get("completed_count", 0),
        }

    except Exception as e:
        logger.exception(f"Exception during combined workflow: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "variants_fasta": str(variants_fasta),
        }
