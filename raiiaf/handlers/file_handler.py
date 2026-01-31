"""RAIIAF file handler for encoding and decoding .raiiaf artifacts.

This module orchestrates packing/unpacking of latent, image, environment, and metadata
chunks, builds the header, and provides a high-level API for reading and writing
RAIIAF files.
"""

import hashlib
from ..core.constants import MAX_FILE_SIZE, MAX_CHUNK_SIZE, HEADER_SIZE, MAX_CHUNKS
from ..core.exceptions import (
    raiiafDecodeError,
    raiiafChunkError,
    raiiafCorruptHeader,
    raiiafImageError,
    raiiafEnvChunkError,
)
from ..core.header import header_init, header_parse, header_validate
from ..chunks.latent import raiiafLatent
from ..chunks.image import raiiafImage
from ..chunks.env import raiiafEnv
from ..chunks.metadata import raiiafMetadata
from typing import Optional, Dict, Any
from PIL import Image
import numpy as np
import struct
import io
import warnings
import json


class raiiafFileHandler:
    """High-level API for reading and writing RAIIAF files."""

    def __init__(self, max_file_size: Optional[int] = None, max_chunk_size: Optional[int] = None):
        """Initialize the file handler with optional size limits.

        Args:
            max_file_size (Optional[int]): Maximum allowed file size in bytes. Defaults to
                MAX_FILE_SIZE.
            max_chunk_size (Optional[int]): Maximum allowed chunk size in bytes. Defaults to
                MAX_CHUNK_SIZE.
        """
        self.latent = raiiafLatent()
        self.image = raiiafImage()
        self.metadata = raiiafMetadata()
        self.env = raiiafEnv()
        self.max_file_size = max_file_size or MAX_FILE_SIZE
        self.max_chunk_size = max_chunk_size or MAX_CHUNK_SIZE
        self.HEADER_SIZE = HEADER_SIZE

    def validate_file_size(self, size: int, context: str = "file") -> bool:
        """Validate a file or chunk size.

        Args:
            size (int): Size in bytes to validate.
            context (str): Validation context; either "file" or "chunk".

        Returns:
            bool: True if the size is within limits.

        Raises:
            raiiafDecodeError: If size exceeds configured limits.
        """
        if size < 0:
            raise raiiafDecodeError(f"Invalid {context} size: {size} (negative)")

        if context == "file" and size > self.max_file_size:
            raise raiiafDecodeError(
                f"File size {size:,} bytes exceeds maximum {self.max_file_size:,} bytes "
                f"({size / (1024**3):.2f} GB)"
            )

        if context == "chunk" and size > self.max_chunk_size:
            raise raiiafChunkError(
                f"Chunk size {size:,} bytes exceeds maximum {self.max_chunk_size:,} bytes"
            )

        return True

    def validate_chunk_count(self, count: int) -> bool:
        """Validate the number of chunks.

        Args:
            count (int): Number of chunks.

        Returns:
            bool: True if the count is within limits.

        Raises:
            raiiafDecodeError: If count exceeds configured limits.
        """
        if count < 0:
            raise raiiafDecodeError(f"Invalid chunk count: {count}")

        if count > MAX_CHUNKS:
            raise raiiafDecodeError(f"Chunk count {count} exceeds maximum {MAX_CHUNKS}")

        return True

    def file_encoder(
        self,
        filename: str,
        latent: Dict[str, np.ndarray],
        chunk_records: list,
        model_name: str,
        model_version: str,
        prompt: str,
        tags: list,
        img_binary: bytes,
        should_compress: bool = True,
        convert_float16: bool = True,
        generation_settings: Optional[dict] = None,
        hardware_info: Optional[dict] = None,
        extra_image: Optional[Dict[str, Any]] = None,
    ):
        """Encode a RAIIAF file.

        Orchestrates packing latent, image, environment, and metadata chunks, builds
        the header, and writes the final file.

        Args:
            filename (str): Output .raiiaf filename. The .raiiaf extension is required.
            latent (Dict[str, np.ndarray]): Mapping of latent keys to arrays.
            chunk_records (list): Mutable list that will be appended with chunk records.
            model_name (str): Name of the model.
            model_version (str): Version of the model.
            prompt (str): Prompt used for generation.
            tags (list): Tags associated with the generation.
            img_binary (bytes): PNG image bytes to embed.
            should_compress (bool): Whether to compress chunks. Defaults to True.
            convert_float16 (bool): Convert latents to float16 for storage. Defaults to True.
            generation_settings (Optional[dict]): Generation configuration to include in metadata.
            hardware_info (Optional[dict]): Hardware information to include in metadata.
            extra_image (Optional[Dict[str, Any]]): Extra fields for the image chunk record.

        Returns:
            dict: A dictionary containing header bytes and chunk bytes with keys:
                - header (bytes)
                - latent_chunks (bytes)
                - metadata_chunk (bytes)
                - image_chunk (Optional[bytes])
        """
        # Pack the latent chunks first and update chunk_records with correct offsets
        if not filename.endswith(".raiiaf"):
            raise ValueError("Filename must have a .raiiaf extension")

        current_offset = self.HEADER_SIZE
        latent_chunks = self.latent.latent_packer(
            latent,
            file_offset=current_offset,
            chunk_records=chunk_records,
            should_compress=should_compress,
            convert_float16=convert_float16,
        )
        latent_chunk = b"".join(latent_chunks)
        self.validate_file_size(len(latent_chunk), "chunk")
        current_offset += len(latent_chunk)

        # Pack image chunk if provided
        image_chunk = None
        if img_binary is not None:
            image_chunk = self.image.image_data_chunk_builder(img_binary)
            image_chunk_record = {
                "type": "DATA",
                "flags": "0000",
                "offset": current_offset,
                "compressed_size": len(image_chunk),
                "uncompressed_size": len(img_binary),
                "hash": hashlib.sha256(img_binary).hexdigest(),
                "extra": extra_image or {},
                "compressed": True,
            }
            chunk_records.append(image_chunk_record)
            current_offset += len(image_chunk)
            print("ENCODER STORED FLAG:", chunk_records[-1]["flags"])

        # Environment chunk
        env_chunk, env_raw = self.env.env_chunk_builder(self.env.env_chunk_populator())
        if len(env_chunk) > 0:
            env_record = {
                "type": "ENVC",
                "flags": "0000",
                "offset": current_offset,
                "compressed_size": len(env_chunk),  # bytes written
                "uncompressed_size": len(env_raw),  # JSON bytes
                "hash": hashlib.sha256(env_raw).hexdigest(),  # hash original
                "extra": {},
                "compressed": True,
            }

            chunk_records.append(env_record)
            current_offset += len(env_chunk)

        # Build manifest with all chunks
        manifest = self.metadata.build_manifest(
            version_major=1,
            version_minor=0,
            model_name=model_name,
            model_version=model_version,
            prompt=prompt,
            tags=tags,
            chunk_records=chunk_records,
            generation_settings=generation_settings,
            hardware_info=hardware_info,
        )
        # Ensure that manifest/metadata is valid
        self.metadata.metadata_validator(manifest)
        # Compress metadata
        compressed_metadata = self.metadata.metadata_compressor(manifest)
        metadata_size = len(compressed_metadata)

        # Calculate total file size
        total_file_size = (
            self.HEADER_SIZE
            + len(latent_chunk)
            + (len(image_chunk) if image_chunk else 0)
            + len(env_chunk)
            + metadata_size
        )
        # Update file_size in manifest
        manifest["raiiaf_metadata"]["file_info"]["file_size"] = total_file_size

        # Recompress metadata with the updated file_size
        compressed_metadata = self.metadata.metadata_compressor(manifest)
        metadata_size = len(compressed_metadata)
        total_file_size = (
            self.HEADER_SIZE
            + len(latent_chunk)
            + (len(image_chunk) if image_chunk else 0)
            + len(env_chunk)
            + len(compressed_metadata)
        )
        chunk_table_offset = (
            self.HEADER_SIZE + len(latent_chunk) + (len(image_chunk) if image_chunk else 0) + len(env_chunk)
        )

        # Final header
        header = header_init(
            version_major=1,
            version_minor=0,
            flags=0,
            chunk_table_offset=chunk_table_offset,
            chunk_table_size=metadata_size,
            chunk_count=len(chunk_records),
            file_size=total_file_size,
        )

        with open(filename, "wb") as f:
            f.write(header)
            f.write(latent_chunk)
            if image_chunk is not None:
                f.write(image_chunk)
            f.write(env_chunk)
            f.write(compressed_metadata)

        return {
            "header": header,
            "latent_chunks": latent_chunk,
            "metadata_chunk": compressed_metadata,
            "image_chunk": image_chunk,
        }

    def file_decoder(self, filename: str):
        """Decode a RAIIAF file.

        Reads the header, chunk table (metadata), and iteratively decodes the
        latent, image, and environment chunks.

        Args:
            filename (str): Path to the input .raiiaf file.

        Returns:
            dict: A dictionary with keys:
                - header (dict): Parsed header fields.
                - chunks (dict): Parsed chunks: 'latent' (list), 'image' (bytes), 'env' (EnvChunk).
                - metadata (dict): Parsed metadata manifest.

        Raises:
            raiiafCorruptHeader: If the header fails validation.
            raiiafChunkError: If a chunk is truncated or corrupt.
            raiiafEnvChunkError: If the environment chunk cannot be parsed when uncompressed.
        """
        with open(filename, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            header = header_parse(header_bytes)
            if not header_validate(header):
                raise raiiafCorruptHeader(message=f"Invalid header: {header}")
            f.seek(header["chunk_table_offset"])
            metadata_compressed = f.read(header["chunk_table_size"])
            metadata = self.metadata.metadata_parser(metadata_compressed)
            chunk_records = metadata["raiiaf_metadata"]["chunks"]
            chunks = {}
            chunks["latent"] = []

            for record in chunk_records:
                chunk_type = record["type"]
                compressed = record.get("compressed", True)

                f.seek(record["offset"])
                raw_chunk = f.read(record["compressed_size"])

                if len(raw_chunk) != record["compressed_size"]:
                    raise raiiafChunkError(
                        f"Truncated chunk {chunk_type} at offset {record['offset']}"
                    )

                if chunk_type == "LATN":
                    shape = tuple(record["extra"]["shape"])

                    if compressed:
                        print("decoded raw_chunk len:", len(raw_chunk))
                        print("expected total bytes:", np.prod(shape))
                        chunk_obj = {
                            "chunk": raw_chunk,
                            "len_header": record.get("len_header"),
                            "len_data": record.get("len_data"),
                        }
                        latent_array = self.latent.latent_parser(chunk_obj, shape, True)
                    else:
                        latent_array = self.latent.latent_parser(raw_chunk, shape, False)

                    chunks["latent"].append(latent_array)
                elif chunk_type == "DATA":
                    if compressed:
                        parsed = self.image.image_data_chunk_parser(raw_chunk)
                        chunks["image"] = parsed["image_data"]
                    else:
                        # DATA chunks have the same header layout even when not compressed
                        chunk_type_b, flags_b, size = struct.unpack("<4s 4s I", raw_chunk[:12])
                        chunks["image"] = raw_chunk[12 : 12 + size]
                elif chunk_type == "ENVC":
                    if compressed:
                        parsed = self.env.env_chunk_parser(raw_chunk)
                        chunks["env"] = parsed["env_chunk"]
                    else:
                        if len(raw_chunk) < 12:
                            raise raiiafEnvChunkError("Truncated ENVC chunk header")
                        chunk_type_b, flags_b, size = struct.unpack("<4s 4s I", raw_chunk[:12])
                        env_json_bytes = raw_chunk[12 : 12 + size]
                        env_dict = json.loads(env_json_bytes.decode("utf-8"))
                        chunks["env"] = env_dict

                    try:
                        # --- Normalize CURRENT environment (from populator) ---
                        current_env_obj = self.env.env_chunk_populator()
                        current_env = {}
                        for comp in current_env_obj.components:
                            if isinstance(comp, dict):
                                comp_id = comp["component_id"]
                                cononical_str = comp["cononical_str"]
                                digest = comp["component_sha256_digest"]
                            else:
                                comp_id = comp.component_id
                                cononical_str = comp.cononical_str  # matches your class
                                digest = comp.component_sha256_digest

                            sha256 = digest.hex() if isinstance(digest, bytes) else digest
                            current_env[comp_id] = {
                                "cononical_str": cononical_str,
                                "sha256": sha256,
                            }

                        # --- Normalize STORED environment (from file) ---
                        stored_raw = chunks["env"]
                        stored_env = {}

                        # Extract components regardless of format
                        if hasattr(stored_raw, "components"):
                            components_list = stored_raw.components
                        elif isinstance(stored_raw, dict) and "components" in stored_raw:
                            components_list = stored_raw["components"]
                        else:
                            components_list = []

                        for comp in components_list:
                            if isinstance(comp, dict):
                                comp_id = comp["component_id"]
                                cononical_str = comp["cononical_str"]
                                digest = comp["component_sha256_digest"]
                            else:
                                comp_id = comp.component_id
                                cononical_str = comp.cononical_str  # ✅
                                digest = comp.component_sha256_digest

                            sha256 = digest.hex() if isinstance(digest, bytes) else digest
                            stored_env[comp_id] = {
                                "cononical_str": cononical_str,
                                "sha256": sha256,
                            }

                        # --- Compare environments ---
                        all_ids = set(stored_env.keys()) | set(current_env.keys())
                        for comp_id in all_ids:
                            stored = stored_env.get(comp_id)
                            current = current_env.get(comp_id)
                            if stored and current:
                                if stored["sha256"] != current["sha256"]:
                                    warnings.warn(
                                        (
                                            f"Environment component '{comp_id}' differs:\n"
                                            f"  File: {stored['cononical_str']}\n"
                                            f"  Current: {current['cononical_str']}"
                                        ),
                                        UserWarning,
                                    )
                            elif stored and not current:
                                warnings.warn(
                                    f"Environment component '{comp_id}' missing in current system",
                                    UserWarning,
                                )
                            elif not stored and current:
                                warnings.warn(
                                    f"Environment component '{comp_id}' missing in file",
                                    UserWarning,
                                )

                    except Exception as e:
                        warnings.warn(f"Failed to compare environment chunks: {e}", UserWarning)
                else:
                    raise ValueError("Unknown chunk type: {chunk_type}. Supported: LATN, DATA")

            return {"header": header, "chunks": chunks, "metadata": metadata}

    @staticmethod
    def png_to_bytes(png_path: str) -> bytes:
        """Convert a PNG image to bytes.

        Preserves transparency by converting to RGBA.

        Args:
            png_path (str): Path to the PNG image.

        Returns:
            bytes: PNG image data in bytes.

        Raises:
            raiiafImageError: If the image cannot be read or encoded.
        """
        try:
            with Image.open(png_path) as img:
                img = img.convert("RGBA")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        except Exception as e:
            raise raiiafImageError(f"Failed to convert PNG to bytes: {e}") from e

    @staticmethod
    def bytes_to_png(img_bytes: bytes) -> Image.Image:
        """Convert PNG bytes to a PIL Image.

        Args:
            img_bytes (bytes): PNG image bytes.

        Returns:
            PIL.Image.Image: Loaded image instance.

        Raises:
            raiiafImageError: If the bytes cannot be decoded as an image.
        """
        try:
            buffer = io.BytesIO(img_bytes)
            img = Image.open(buffer)
            return img
        except Exception as e:
            raise raiiafImageError(f"Failed to convert bytes to PNG: {e}") from e
