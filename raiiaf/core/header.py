"""Header utilities for RAIIAF files.

This module defines functions to initialize, parse, and validate the RAIIAF
file header. The header layout is defined by ``raiiaf.core.constants.HEADER_FORMAT``
and has the following fields:

- magic (6s): ASCII bytes b"raiiaf"
- version_major (B)
- version_minor (B)
- flags (B)
- chunk_table_offset (I)
- chunk_table_size (I)
- chunk_count (I)
- file_size (I)
- reserved (I)
"""

import struct
from .constants import HEADER_SIZE, HEADER_FORMAT
from .exceptions import raiiafCorruptHeader


def header_init(
    version_major=1,
    version_minor=0,
    flags=0,
    chunk_table_offset=0,
    chunk_table_size=0,
    chunk_count=0,
    file_size=0,
    reserved=0,
):
    """Initialize and pack a RAIIAF file header.

    Args:
        version_major (int): Major version number.
        version_minor (int): Minor version number.
        flags (int): Header flags.
        chunk_table_offset (int): Offset (in bytes) to the start of the chunk table.
        chunk_table_size (int): Size (in bytes) of the chunk table.
        chunk_count (int): Number of chunks referenced in the table.
        file_size (int): Total file size in bytes.
        reserved (int): Reserved for future use.

    Returns:
        bytes: Packed header bytes conforming to HEADER_FORMAT.
    """
    return struct.pack(
        HEADER_FORMAT,
        b"raiiaf",
        version_major,
        version_minor,
        flags,
        chunk_table_offset,
        chunk_table_size,
        chunk_count,
        file_size,
        reserved,
    )


def header_parse(buf: bytes) -> dict:
    """Parse a RAIIAF header from raw bytes.

    Args:
        buf (bytes): Raw file bytes containing the header at the beginning.

    Returns:
        dict: Parsed header fields with keys: magic, version_major, version_minor,
            flags, chunk_table_offset, chunk_table_size, chunk_count, file_size, reserved.

    Raises:
        raiiafCorruptHeader: If the provided buffer is too small.
    """
    import struct

    if len(buf) < HEADER_SIZE:
        raise raiiafCorruptHeader(message="Header too small")
    vals = struct.unpack(HEADER_FORMAT, buf[:HEADER_SIZE])
    magic = vals[0]
    version_major = vals[1]
    version_minor = vals[2]
    flags = vals[3]
    chunk_table_offset = vals[4]
    chunk_table_size = vals[5]
    chunk_count = vals[6]
    file_size = vals[7]
    reserved = vals[8]
    return {
        "magic": magic,
        "version_major": version_major,
        "version_minor": version_minor,
        "flags": flags,
        "chunk_table_offset": chunk_table_offset,
        "chunk_table_size": chunk_table_size,
        "chunk_count": chunk_count,
        "file_size": file_size,
        "reserved": reserved,
    }


def header_validate(header: dict) -> bool:
    """Validate a parsed RAIIAF header.

    Args:
        header (dict): Parsed header fields as returned by header_parse.

    Returns:
        bool: True if the header is valid, False otherwise.
    """
    if header["magic"] != b"raiiaf":
        return False
    if header["version_major"] < 1:
        return False
    if header["chunk_count"] < 0:
        return False
    return True
