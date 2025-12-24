"""Shared utility functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union


def count_sequences_in_a3m(a3m_path: Union[str, Path]) -> int:
    """Count the number of sequences in the A3M file.

    Extracted from examples/use_case_3_precomputed_msa.py.

    Args:
        a3m_path: Path to the A3M file

    Returns:
        Number of sequences in the alignment
    """
    count = 0
    try:
        with open(a3m_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
    except Exception:
        count = 0
    return count


def calculate_relative_path(source_path: Path, reference_dir: Path) -> str:
    """Calculate relative path from reference directory to source file.

    Extracted from examples/use_case_3_precomputed_msa.py.

    Args:
        source_path: Path to the source file
        reference_dir: Reference directory to calculate relative path from

    Returns:
        Relative path as string, or absolute path if not possible
    """
    try:
        source_abs = source_path.resolve()
        reference_abs = reference_dir.resolve()
        relative_path = source_abs.relative_to(reference_abs)
        return str(relative_path)
    except ValueError:
        # Files are not on the same relative path, use absolute path
        return str(source_abs)