"""Shared library for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
All functions use only Python standard library.
"""

from .io import (
    read_fasta,
    save_json,
    load_json,
)

from .validation import (
    validate_protein_sequence,
    validate_smiles,
    validate_a3m_file,
)

from .utils import (
    count_sequences_in_a3m,
    calculate_relative_path,
)

__all__ = [
    # I/O functions
    "read_fasta",
    "save_json",
    "load_json",
    # Validation functions
    "validate_protein_sequence",
    "validate_smiles",
    "validate_a3m_file",
    # Utility functions
    "count_sequences_in_a3m",
    "calculate_relative_path",
]