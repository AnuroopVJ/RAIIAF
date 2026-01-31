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

def test_corrupt_metadata(tmp_path):
    filename = tmp_path / "corrupt.raiiaf"
    latent = {"latent_1": torch.randn(1, 4, 64, 64).numpy()}
    img_bytes = create_test_image()

    raiiaf.file_encoder(
        filename=str(filename),
        latent=latent,
        chunk_records=[],
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
            "os": "windows",
            "cpu": "Intel Core i7",
            "cpu_cores": 8,
            "gpu": [],
            "ram_gb": 16.0,
            "framework": "torch",
            "compute_lib": "cpu"
        }
    )


    with open(filename, "rb") as f:
        header_bytes = f.read(raiiaf.HEADER_SIZE)
        header = header_parse(header_bytes)

    with open(filename, "r+b") as f:
        f.seek(header['chunk_table_offset'])
        f.write(b"\x00" * min(100, header['chunk_table_size']))

    with pytest.raises((raiiafMetadataError, json.JSONDecodeError, zstd.ZstdError)):
        raiiaf.file_decoder(str(filename))
