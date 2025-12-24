"""Shared validation functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union


def validate_protein_sequence(sequence: str) -> bool:
    """Validate that the sequence contains only valid amino acids.

    Extracted from examples/use_case_*.py files.

    Args:
        sequence: Amino acid sequence

    Returns:
        True if valid, False otherwise
    """
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    return all(aa.upper() in valid_aa for aa in sequence)


def validate_smiles(smiles: str) -> bool:
    """Basic validation of SMILES string (check for common characters).

    Extracted from examples/use_case_2_protein_ligand.py.

    Args:
        smiles: SMILES string

    Returns:
        True if appears valid, False otherwise
    """
    valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]@+=\\/#-')
    return len(smiles) > 0 and all(c in valid_chars for c in smiles)


def validate_a3m_file(a3m_path: Union[str, Path]) -> tuple[bool, str]:
    """Validate that the A3M file exists and has basic A3M format.

    Extracted from examples/use_case_3_precomputed_msa.py.

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