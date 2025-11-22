"""
Hash validation helper for code integrity checks.
Uses CRC32 for fast hash computation to prevent tampering with NSFW detection.
"""

import os
import zlib
from typing import Optional


def create_hash(content: bytes) -> str:
    """Create CRC32 hash of content."""
    return format(zlib.crc32(content), '08x')


def validate_hash(validate_path: str) -> bool:
    """
    Validate a file against its .hash file.
    
    Args:
        validate_path: Path to file to validate
        
    Returns:
        True if hash matches, False otherwise
    """
    hash_path = get_hash_path(validate_path)
    
    if hash_path and os.path.isfile(hash_path):
        try:
            with open(hash_path, 'r') as hash_file:
                expected_hash = hash_file.read().strip()
            
            with open(validate_path, 'rb') as validate_file:
                content = validate_file.read()
            
            actual_hash = create_hash(content)
            return actual_hash == expected_hash
        except Exception:
            return False
    return False


def get_hash_path(validate_path: str) -> Optional[str]:
    """Get the .hash file path for a given file."""
    if os.path.isfile(validate_path):
        directory, filename = os.path.split(validate_path)
        name, _ = os.path.splitext(filename)
        return os.path.join(directory, name + '.hash')
    return None


def validate_module_integrity(module_path: str, expected_hash: str) -> bool:
    """
    Validate Python module hasn't been tampered with.
    
    Args:
        module_path: Path to .py file
        expected_hash: Expected CRC32 hash
        
    Returns:
        True if hash matches
    """
    if not os.path.isfile(module_path):
        return False
    
    try:
        with open(module_path, 'rb') as f:
            content = f.read()
        actual_hash = create_hash(content)
        return actual_hash == expected_hash
    except Exception:
        return False

