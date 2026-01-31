"""Environment chunk utilities for RAIIAF.

Defines dataclasses for environment components and provides helpers to populate,
build (compress), and parse environment (ENVC) chunks.
"""

import numpy as np
import sys
import hashlib
from raiiaf.core.exceptions import raiiafEnvChunkError
from dataclasses import dataclass, asdict
from typing import List
import torch
import json
import zstandard as zstd
import pynvml as nvml
import struct
import zstandard as zstd
import platform


@dataclass
class EnvComponent:
    """Single environment component captured in the ENVC chunk.

    Attributes:
        component_id (str): Identifier (e.g., 'torch', 'numpy', 'python', 'cuda', 'os', 'gpu').
        cononical_str (str): Canonical string describing the component and version.
        component_sha256_digest (bytes): SHA-256 digest for the canonical string.
    """

    component_id: str
    cononical_str: str
    component_sha256_digest: bytes


@dataclass
class EnvChunk:
    """Structured environment chunk content.

    Attributes:
        env_version (int): Environment chunk version number.
        components (List[EnvComponent]): List of environment components.
    """

    env_version: int
    components: List[EnvComponent]


class raiiafEnv:
    """Operations for RAIIAF environment (ENVC) chunks."""

    def __init__(self):
        pass

    def env_chunk_populator(self):
        """Populate the environment chunk with component hashes.

        Returns:
            EnvChunk: Populated environment chunk object with common components.
        """
        components = []
        # torch
        torch_version = torch.__version__.split("+")[0]
        cuda_version = torch.version.cuda
        cononical_str = f"name=torch;version={torch_version};cuda={cuda_version}"
        sha256_digest = hashlib.sha256(cononical_str.encode("utf-8")).digest()
        torch_component = EnvComponent(
            component_id="torch", cononical_str=cononical_str, component_sha256_digest=sha256_digest
        )
        components.append(torch_component)
        # numpy
        numpy_version = np.__version__
        cononical_str = f"name=numpy;version={numpy_version}"
        sha256_digest = hashlib.sha256(cononical_str.encode("utf-8")).digest()
        numpy_component = EnvComponent(
            component_id="numpy", cononical_str=cononical_str, component_sha256_digest=sha256_digest
        )
        components.append(numpy_component)
        # python
        python_version = platform.python_version()
        cononical_str = f"name=python;version={python_version}"
        sha256_digest = hashlib.sha256(cononical_str.encode("utf-8")).digest()
        python_component = EnvComponent(
            component_id="python", cononical_str=cononical_str, component_sha256_digest=sha256_digest
        )
        components.append(python_component)
        # CUDA
        cuda_version = torch.version.cuda
        if cuda_version is None:
            cuda_version = "none"
        cononical_str = f"name=cuda;version={cuda_version}"
        sha256_digest = hashlib.sha256(cononical_str.encode("utf-8")).digest()
        cuda_component = EnvComponent(
            component_id="cuda", cononical_str=cononical_str, component_sha256_digest=sha256_digest
        )
        components.append(cuda_component)
        # OS
        os_version = f"{platform.system()} {platform.release()}"
        cononical_str = f"name=os;version={os_version}"
        sha256_digest = hashlib.sha256(cononical_str.encode("utf-8")).digest()
        os_component = EnvComponent(
            component_id="os", cononical_str=cononical_str, component_sha256_digest=sha256_digest
        )
        components.append(os_component)
        # GPU
        try:
            # attempt only if cuda is available
            if not torch.cuda.is_available():
                nvml.nvmlInit()
                try:
                    gpu_count = nvml.nvmlDeviceGetCount()
                    driver_version = nvml.nvmlSystemGetDriverVersion().decode("utf-8")
                    for i in range(gpu_count):
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
                        cononical_str = f"name=gpu;model={name};driver={driver_version}"
                        sha256_digest = hashlib.sha256(cononical_str.encode("utf-8")).digest()
                        gpu_component = EnvComponent(
                            component_id="gpu",
                            cononical_str=cononical_str,
                            component_sha256_digest=sha256_digest,
                        )
                        components.append(gpu_component)
                finally:
                    nvml.nvmlShutdown()  # to avoid resource leaks
        except Exception:
            pass

        return EnvChunk(env_version=1, components=components)

    def env_chunk_builder(self, env_chunk: EnvChunk):
        """Build a compressed ENVC chunk from an EnvChunk object.

        Args:
            env_chunk (EnvChunk): Populated environment chunk object.

        Returns:
            Tuple[bytes, bytes]: Compressed chunk bytes and the raw env JSON bytes.

        Raises:
            raiiafEnvChunkError: If building or compression fails.
        """
        try:
            env_dict = asdict(env_chunk)

            # convert bytes to hex in each of the components
            for comp in env_dict["components"]:
                if isinstance(comp["component_sha256_digest"], bytes):
                    comp["component_sha256_digest"] = comp["component_sha256_digest"].hex()

            env_json = json.dumps(env_dict, indent=2)
            env_bytes = env_json.encode("utf-8")
            chunk_type = b"ENVC"
            chunk_flags = b"0000"
            chunk_size = len(env_bytes)
            header = struct.pack("<4s 4s I", chunk_type, chunk_flags, chunk_size)
            full_chunk = header + env_bytes
            compressor = zstd.ZstdCompressor()
            compressed = compressor.compress(full_chunk)

            return compressed, env_bytes
        except Exception as e:
            raise raiiafEnvChunkError(f"Failed to build environment chunk: {e}") from e

    def env_chunk_parser(self, compressed_chunk):
        """Parse a compressed ENVC chunk.

        Args:
            compressed_chunk (bytes): Compressed environment chunk.

        Returns:
            dict: Parsed environment with keys 'chunk_type', 'chunk_flags', 'chunk_size', 'env_chunk'.
        """
        decompressor = zstd.ZstdDecompressor()
        chunk = decompressor.decompress(compressed_chunk)
        chunk_type, chunk_flags, chunk_size = struct.unpack("<4s 4s I", chunk[:12])
        env_chunk_bytes = chunk[12 : 12 + chunk_size]
        env_chunk_json = env_chunk_bytes.decode("utf-8")
        env_chunk_dict = json.loads(env_chunk_json)
        for comp in env_chunk_dict["components"]:
            digest = comp["component_sha256_digest"]
            if isinstance(digest, str) and len(digest) == 64:
                try:
                    comp["component_sha256_digest"] = bytes.fromhex(digest)
                except ValueError:
                    pass

        env_chunk = EnvChunk(**env_chunk_dict)
        return {
            "chunk_type": chunk_type,
            "chunk_flags": chunk_flags,
            "chunk_size": chunk_size,
            "env_chunk": env_chunk,
        }
