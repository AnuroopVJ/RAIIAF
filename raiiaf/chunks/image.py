"""Image chunk utilities for RAIIAF.

Provides validation, packing (compression) and parsing for image data chunks
(DATA). Uses zstd for compression.
"""

import io
from PIL import Image
from zstandard import ZstdCompressor, ZstdDecompressor
from ..core.exceptions import raiiafImageError
import struct


class raiiafImage:
    """Operations for RAIIAF image (DATA) chunks."""

    def __init__(self):
        pass

    def image_bytes_validator(self, image_bytes: bytes):
        """Validate image bytes.

        Args:
            image_bytes (bytes): Image bytes to validate.

        Returns:
            bool: True if the image bytes form a valid image.

        Raises:
            raiiafImageError: If validation fails.
        """
        try:
            img = io.BytesIO(image_bytes)
            with Image.open(img) as img:
                img.verify()
            return True
        except Exception as e:
            raise raiiafImageError(f"Invalid image bytes: {e}")

    def image_data_chunk_builder(self, image_binary: bytes):
        """Build a compressed image DATA chunk from raw image bytes.

        Args:
            image_binary (bytes): Raw PNG image bytes to store.

        Returns:
            bytes: Compressed chunk bytes suitable for writing to the RAIIAF file.

        Raises:
            raiiafImageError: If the chunk cannot be constructed.
        """
        self.image_bytes_validator(image_binary)
        try:
            chunk_type = b"DATA"
            chunk_flags = b"0000"
            chunk_size = len(image_binary)
            chunk_header = struct.pack("<4s 4s I", chunk_type, chunk_flags, chunk_size)
            chunk = chunk_header + image_binary
            compressed_chunk = ZstdCompressor().compress(chunk)
            return compressed_chunk
        except Exception as e:
            raise raiiafImageError(f"Failed to build image data chunk: {e}") from e

    def image_data_chunk_parser(self, compressed_chunk):
        """Parse a compressed image DATA chunk.

        Args:
            compressed_chunk (bytes): Compressed image data chunk.

        Returns:
            dict: Parsed info with keys 'chunk_type', 'chunk_flags', 'chunk_size', 'image_data'.

        Raises:
            raiiafImageError: If decompression or parsing fails.
        """
        try:
            decompressor = ZstdDecompressor()
            chunk = decompressor.decompress(compressed_chunk)
            chunk_type, chunk_flags, chunk_size = struct.unpack("<4s 4s I", chunk[:12])
            image_data = chunk[12 : 12 + chunk_size]
            return {
                "chunk_type": chunk_type,
                "chunk_flags": chunk_flags,
                "chunk_size": chunk_size,
                "image_data": image_data,
            }
        except Exception as e:
            raise raiiafImageError(f"Failed to parse image data chunk: {e}") from e
