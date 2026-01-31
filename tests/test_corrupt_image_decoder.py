from pathlib import Path
import numpy as np
import tempfile
import os
import torch
import pytest
import json.decoder
import zstandard as zstd
import copy
from raiiaf import raiiafFileHandler
from raiiaf.core.exceptions import raiiafCorruptHeader, raiiafMetadataError, raiiafImageError, raiiafChunkError
from raiiaf.chunks.metadata import raiiafMetadata
from raiiaf.core.header import header_parse
from PIL import Image
import io


raiiaf = raiiafFileHandler()
def create_test_image():
    img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 255))  #justa  red square
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_corrupt_image_decoder():
    img_bytes = create_test_image()
    latent = {
        "latent_1": torch.randn(1, 4, 64, 64, dtype=torch.float32).numpy()
    }
    chunk_records = []

    with tempfile.NamedTemporaryFile(suffix=".raiiaf", delete=False) as tmp_file:
        filename = tmp_file.name

    result = raiiaf.file_encoder(
    filename=filename,
    latent=latent,
    chunk_records=chunk_records,
    model_name="TestModel",
    model_version="1.0",
    prompt="Test prompt",
    tags=["test"],
    img_binary=img_bytes,
    convert_float16=False,
    generation_settings={
        "seed": 42,
        "steps": 20,
        "sampler": "ddim",
        "cfg_scale": 7.5,
        "scheduler": "pndm",
        "eta": 0.0,
        "guidance": "classifier-free",
        "precision": "fp16",
        "deterministic": True
    },
    hardware_info={
        "machine_name": "test_machine",
        "os": "linux",
        "cpu": "Intel",
        "cpu_cores": 8,
        "gpu": [{"name": "RTX 3090", "memory_gb": 24, "driver": "nvidia", "cuda_version": "12.1"}],
        "ram_gb": 64.0,
        "framework": "torch",
        "compute_lib": "cuda"
    }
)
    metadata = raiiafMetadata().metadata_parser(result["metadata_chunk"])
    with open(filename, "r+b") as f:
        for rec in metadata["raiiaf_metadata"]["chunks"]:
            if rec["type"] == "DATA":          #image chunk
                f.seek(rec["offset"])
                f.write(b"\xFF" * rec["compressed_size"])
                break

    with pytest.raises(raiiafImageError):
        raiiaf.file_decoder(filename)