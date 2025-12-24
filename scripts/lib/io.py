"""Shared I/O functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Any, Dict
import json


def read_fasta(file_path: Union[str, Path]) -> tuple[str, str]:
    """Read a single-sequence FASTA file.

    Extracted from examples/use_case_*.py files.

    Args:
        file_path: Path to the FASTA file

    Returns:
        Tuple of (sequence_name, sequence)

    Raises:
        ValueError: If file is empty or has invalid format
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {file_path}")

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Empty FASTA file")

    if not lines[0].startswith('>'):
        raise ValueError("Invalid FASTA format - missing header")

    name = lines[0][1:]  # Remove '>' prefix
    sequence = ''.join(lines[1:])  # Join all sequence lines

    return name, sequence


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save the JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file has invalid JSON
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, 'r') as f:
        return json.load(f)