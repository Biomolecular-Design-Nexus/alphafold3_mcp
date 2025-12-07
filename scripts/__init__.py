"""AlphaFold3 Python Scripts Module.

This module provides Python interfaces for running AlphaFold3 structure predictions
and preparing variant configurations.

Prediction Functions:
    - run_default: Full MSA search + template search + inference
    - run_with_a3m: Pre-computed A3M MSA + optional template search + inference
    - run_with_msa: Pre-computed MSA/templates (inference only, fastest)
    - run_batch: Batch processing for multiple variants

Variant Preparation Functions:
    - prepare_variant_configs: Prepare input configs for multiple variants
    - parse_fasta: Parse FASTA file to get variant sequences
    - load_wt_data: Load wild-type data from AF3 output JSON
    - create_variant_input: Create input JSON for a single variant

Example Usage:
    # Run default full pipeline
    from scripts import run_default
    results = run_default("/path/to/data", device=0)

    # Run with pre-computed A3M
    from scripts import run_with_a3m
    result = run_with_a3m("/path/to/data", run_template_search=True)

    # Batch processing
    from scripts import run_batch
    result = run_batch("/path/to/variants", skip_existing=True)

    # Prepare variant configs
    from scripts import prepare_variant_configs
    result = prepare_variant_configs(
        variants_fasta="variants.fasta",
        wt_data_json="wt_data.json",
        output_dir="variants",
    )

    # Full workflow: prepare variants and run predictions
    from scripts import prepare_and_run_variants
    result = prepare_and_run_variants(
        variants_fasta="variants.fasta",
        wt_data_json="wt_data.json",
        output_dir="variants",
    )
"""

# Import prediction functions
from .alphafold3_runner import (
    run_default,
    run_with_a3m,
    run_with_msa,
    run_batch,
    # Aliases
    predict_structure,
    predict_with_msa,
    predict_msa_only,
    predict_batch,
    # Result class
    PredictionResult,
    # Logging utilities
    set_log_level,
    add_file_logger,
)

# Import variant preparation functions
from .prepare_variants import (
    prepare_variant_configs,
    parse_fasta,
    load_wt_data,
    create_variant_input,
    create_variant_msa,
    prepare_and_run_variants,
    # Data classes
    VariantInfo,
    WildTypeData,
    # Logging utilities
    set_log_level as set_variant_log_level,
)

__all__ = [
    # Prediction functions
    "run_default",
    "run_with_a3m",
    "run_with_msa",
    "run_batch",
    # Prediction aliases
    "predict_structure",
    "predict_with_msa",
    "predict_msa_only",
    "predict_batch",
    # Variant preparation functions
    "prepare_variant_configs",
    "parse_fasta",
    "load_wt_data",
    "create_variant_input",
    "create_variant_msa",
    "prepare_and_run_variants",
    # Data classes
    "PredictionResult",
    "VariantInfo",
    "WildTypeData",
    # Logging utilities
    "set_log_level",
    "add_file_logger",
    "set_variant_log_level",
]

__version__ = "0.1.0"
