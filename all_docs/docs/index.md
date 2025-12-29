# Welcome to Gen5 File Format Documentation

For the source code visit [github](https://github.com/AnuroopVJ/gen5)

## Overview

Gen5 is a binary container format aimed at increased reproducibility for AI-generated images. It enables the storage of several key pieces of information, such as :

- The initial noise tensor (which usually changes every run)
- Model name and version
- Prompt  
- Tags  
- Hardware information  
- Generation settings
(may include sampler-specific parameters)

The Initial noise tensor can be fed back in while using a model (local ones) to obtain similar results.
This has proven to be capable of producing extremely similar images. Although we use a random seed integer value, the usage of the real tensor provides increased reproducibility.


## Installation
Just pip install the package!
```bash
pip install gen5
```
## Usage
import the classes
```python
from gen5.main import Gen5FileHandler
```
First you need to instantiate the Gen5FileHandler class.
```python
gen5 = Gen5FileHandler()
```

# Encoding
!!! danger
    **DISCLAIMER**:
    The encoder expects **NumPy arrays**.  
    If you use PyTorch tensors, convert them with `.detach().cpu().numpy()`.

```python
from gen5.main import Gen5FileHandler

gen5 = Gen5FileHandler()
initial_noise_tensor = torch.randn(batch_size, channels, height, width)
latent = {
    "initial_noise": initial_noise_tensor.detach().cpu().numpy() #The encoder expects numpy array not a torch tensor object
}
binary_img_data = gen5.png_to_bytes(r'path/to/image.png') # use the helper function to convert image to bytes

gen5.file_encoder(
    filename="encoded_img.gen5", # The .gen5 extension is required!
    latent=latent,# initial latent noise
    chunk_records=[],
    model_name="Stable Diffusion 3",
    model_version="3", # Model Version
    prompt="A puppy smiling, cinematic",
    tags=["puppy","dog","smile"],
    img_binary=binary_img_data,
    convert_float16=False, # whether to convert to float16 (enable if input tensors is in float32)
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
        "cpu_cores": 8, # minimum 1
        "gpu": [{"name": "RTX 3090", "memory_gb": 24, "driver": "nvidia", "cuda_version": "12.1"}],
        "ram_gb": 64.0,
        "framework": "torch",
        "compute_lib": "cuda"
    }
)
```

# Decoding
```python
decoded = gen5.file_decoder(filename)
# Now to save the metadata
metadata = decoded["metadata"]["gen5_metadata"]

# to just get specific metadata blocks
model_info = decoded["metadata"]["gen5_metadata"]["model_info"]

# to save decoded metadata to a json file
with open("decoded_metadata.json", "w") as f:
    json.dump(decoded["metadata"], f, indent=2)

# to save just the image_binary as png
image_bytes = decoded["chunks"].get("image")
if image_bytes is not None:
    img = Image.open(io.BytesIO(image_bytes))
    img.save("decoded_image.png")
```

## Examples
```python
from gen5.main import Gen5FileHandler
import torch
import json
import numpy as np
import io
from PIL import Image
gen5 = Gen5FileHandler()

batch_size = 1
channels = 3  # For RGB images
height = 64
width = 64

# Generate the initial noise tensor (often called z_T or x_T)
initial_noise_tensor = torch.randn(batch_size, channels, height, width)
binary_img_data = gen5.png_to_bytes(r"C:\Users\neela\Desktop\Miscellaneous\image file format - .gen5\gen5\src\gen5\example.png")
latent = {
    "initial_noise": initial_noise_tensor.detach().cpu().numpy() # The encoder expects numpy array not a torch tensor object
}
gen5.file_encoder(
    filename="converted_img.gen5",
    latent=latent,
    chunk_records=[],
    model_name="Stable Diffusion 3",
    model_version="3",
    prompt="A puppy smiling, cinematic",
    tags=["puppy","dog","smile"],
    img_binary=binary_img_data,
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
print("Image Encoded Successfully...")
decoded = gen5.file_decoder(
    r"path/to/decoded.gen5"
)

with open("decoded_metadata.json", "w") as f:
    json.dump(decoded["metadata"], f, indent=2)

image_bytes = decoded["chunks"].get("image")
if image_bytes is not None:
    img = Image.open(io.BytesIO(image_bytes))
    img.save("decoded_image.png")

latent_data = decoded["chunks"].get("latent", [])
for i, latent_array in enumerate(latent_data):
    np.save(f"latent_{i}.npy", latent_array)
print("Decoded metadata saved to decoded_metadata.json")
if image_bytes is not None:
    print("Decoded image saved to decoded_image.png")
```
Expected decoded metadata json:
```json
{
  "gen5_metadata": {
    "file_info": {
      "magic": "GEN5",
      "version_major": 1,
      "version_minor": 0,
      "file_size": 67786,
      "chunk_count": 2
    },
    "model_info": {
      "model_name": "Stable Diffusion 3",
      "version": "3",
      "date": "2025-12-29T03:22:11.273315+00:00",
      "prompt": "A puppy smiling, cinematic",
      "tags": [
        "puppy",
        "dog",
        "smile"
      ],
      "generation_settings": {
        "seed": 42,
        "steps": 20,
        "sampler": "ddim",
        "cfg_scale": 7.5,
        "scheduler": "pndm",
        "eta": 0.0,
        "guidance": "classifier-free",
        "precision": "fp16",
        "deterministic": true
      },
      "hardware_info": {
        "machine_name": "test_machine",
        "os": "linux",
        "cpu": "Intel",
        "cpu_cores": 8,
        "gpu": [
          {
            "name": "RTX 3090",
            "memory_gb": 24,
            "driver": "nvidia",
            "cuda_version": "12.1"
          }
        ],
        "ram_gb": 64.0,
        "framework": "torch",
        "compute_lib": "cuda"
      }
    },
    "chunks": [
      {
        "index": 0,
        "type": "LATN",
        "flags": "F32",
        "offset": 32,
        "compressed_size": 45436,
        "uncompressed_size": 49152,
        "hash": "09ab27d5354d70d0845832e23026d78565b8645533198e6d523b0c5485e8d94e",
        "extra": {
          "shape": [
            1,
            3,
            64,
            64
          ],
          "dtype": "float32"
        },
        "compressed": true
      },
      {
        "index": 1,
        "type": "DATA",
        "flags": "0000",
        "offset": 45468,
        "compressed_size": 21500,
        "uncompressed_size": 24744,
        "hash": "7f81bf1105c4dbadec1dc137a5e7106332b0b25e5425106b856833348e9874ae",
        "extra": {},
        "compressed": true
      }
    ]
  }
}
```


If you need further help, feel free to [contact me!](https://mail.google.com/mail/?view=cm&fs=1&to=avjisauser@gmail.com)