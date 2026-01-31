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

def test_decoder_bad_magic(tmp_path: Path):

    filename = tmp_path / "test.raiiaf"
    batch_size = 1
    channels = 4
    height = 64
    width = 64
    initial_noise_latent = {
        "latent_1": torch.randn(batch_size, channels, height, width, dtype=torch.float32).numpy()
    }
    chunk_records = []
    img_bytes = create_test_image()
    raiiaf.file_encoder(
        filename=str(filename),
        latent=initial_noise_latent,
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


    # Corrupt header
    with open(filename, "r+b") as f:
        f.write(b"XXXX")

    with pytest.raises(raiiafCorruptHeader):
        raiiaf.file_decoder(str(filename))