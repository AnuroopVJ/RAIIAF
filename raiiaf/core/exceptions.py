"""Exception hierarchy for RAIIAF operations.

Defines specialized exceptions for decode errors, corrupt headers, metadata issues,
chunk-level errors, and specific latent/image/environment chunk problems.
"""

class raiiafDecodeError(Exception):
    """Base error for RAIIAF decoding/validation failures."""

    def __init__(self, message: str):
        super().__init__(f"raiiaf Decode Error: {message}")


class raiiafCorruptHeader(raiiafDecodeError):
    """Raised when the file header is detected as corrupt or invalid."""

    def __init__(self, message: str):
        super().__init__(f"Corrupt header: {message}")


class raiiafMetadataError(raiiafDecodeError):
    """Raised when the metadata (manifest) is invalid or cannot be parsed."""

    def __init__(self, message: str):
        super().__init__(f"Corrupt Metadata: {message}")


class raiiafChunkError(raiiafDecodeError):
    """Raised for generic chunk-level issues (truncation, bounds, etc.)."""

    def __init__(self, message: str):
        super().__init__(f"Corrupt Chunk: {message}")


class raiiafLatentError(raiiafChunkError):
    """Raised for latent (LATN) chunk issues."""

    def __init__(self, message: str):
        super().__init__(f"Corrupt Latent: {message}")


class raiiafImageError(raiiafChunkError):
    """Raised for image (DATA) chunk issues or image conversion errors."""

    def __init__(self, message: str):
        super().__init__(f"Corrupt Image: {message}")


class raiiafEnvChunkError(raiiafChunkError):
    """Raised for environment (ENVC) chunk parsing/validation issues."""

    def __init__(self, message: str):
        super().__init__(f"Corrupt Environment Chunk: {message}")
