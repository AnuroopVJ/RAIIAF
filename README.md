# .gen5 - AI-native Context Storage

![bannerimage](.gen_file_format.png)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyPI](https://img.shields.io/pypi/v/gen5)

## Features
- Noise latent tensor storage
- Rich AI-native metadata that includes:
   
      - Model name and version
      - Prompt  
      - Tags  
      - Hardware information  
      - Generation settings
- Environment info stored automatically as hashed canonical strings
- Issues warnings if environment drift is detected!

# Why not just use EXIF/XMP/Sidecar files?
Here is an emperical comparison:

| Aspect | EXIF / XMP (Custom Metadata) | GEN5 |
|------|-------------------------------|------|
| Schema enforcement | Convention-based, unenforced | 🟢 Canonical, versioned schema |
| Semantic consistency | Low; tag drift common | 🟢 High; fixed fields + chunk types |
| Latent representation | Not supported (text hacks) | 🟢 Native latent chunks (binary-safe) |
| Environment capture | Ad-hoc text notes | 🟢 Explicit env chunks (model, seed, hw, libs) |
| Reproducibility ceiling | Limited, state incomplete | 🟢 High; full generation state captured |
| Data typing | Weak (string-heavy XML) | 🟢 Strong typing (binary, arrays, structs) |
| Extensibility | Easy but uncontrolled | 🟡 Controlled; safer but slower evolution |
| Tooling ecosystem | Mature, ubiquitous | 🔴 Immature, GEN5-specific tools needed |
| Interoperability | Works almost everywhere | 🔴 Breaks without GEN5-aware readers |
| Failure mode | Metadata silently ignored | 🟡 Metadata explicit but unreadable without tooling |

Although XMP has the ability to embed any binary payload, doing so necessitates ad hoc conventions, manual validation, and cautious handling to prevent silent data loss. By formalizing these practices into a first-class, schema-enforced representation, GEN5 lessens failure modes and implementation burden.

a comparison with sidecar files: 

| Aspect | Where Sidecar (XMP) shines / empirical behavior | Where GEN5 shines / empirical behavior |
|---|---|---|
| Non-destructive editing | 🟢 Sidecars allow metadata edits without touching the original asset (important for read-only/ RAW workflows). Many DAMs expose "write to sidecar only" modes. | 🟡 GEN5 typically embeds state into a container; editing metadata might require GEN5-aware tooling. Good for integrity but less “drop-in” non-destructive edit unless you design a sidecar-style GEN5 wrapper. |
| Support & tooling (read/write) | 🟢 Very broad: exiftool, Lightroom, digikam and many tools understand XMP and sidecars; exiftool can create and sync sidecars. This is production proven.| 🔴 Immature: GEN5 requires custom readers/writers; integration work required. This is the main adoption barrier (engineering cost). |
| Round-trip fidelity (read->modify->write->read) | 🔴 Variable: many apps will normalize/sync/overwrite XMP fields; different tools may not round-trip custom blobs reliably (risk of normalization or loss). Empirical reports of inconsistent XMP syncing and sidecar issues exist. | 🟢 High if tools are spec-compliant: GEN5’s schema + chunk validation lets compliant tools preserve unknown chunks and guarantee round-trip fidelity (design goal). |
| Orphaning & file management | 🔴 Sidecars can be orphaned (moved, renamed, or not uploaded); cloud/backup processes often miss them — real world bug reports and GH issues. | 🟢 GEN5 embeds state into the artifact (single file), eliminating the orphaning problem — better for long-term datasets and archives. |
| Resilience to platform stripping | 🔴 Both can be stripped in public pipelines; embedded XMP is sometimes removed by social platforms and services. Sidecars are even more fragile because many uploaders ignore sidecar files. | 🟡 GEN5 helps when you control the pipeline (archives, datasets). For public publishing (social platforms) nothing is guaranteed unless the platform preserves custom blocks — but embedding reduces the chance of separate-file loss. |
| Validation & semantics | 🔴 XMP allows arbitrary namespaces; no enforcement — different users/tools will store semantically identical things under different keys (fragmentation). Empirical evidence of inconsistent metadata across tools.| 🟢 GEN5 enforces typed schema, versioning, and chunk semantics → machine-verifiable metadata (reduces semantic drift). This is the core reproducibility gain. |
| Indexing & large-scale querying | 🟡 Sidecars can be indexed when present, but inconsistent field names and scattered sidecars complicate large dataset indexing.| 🟢 GEN5’s structured schema makes programmatic indexing and deterministic query semantics straightforward — ideal for datasets and benchmarks. |
| Forensic reproducibility (latents + env) | 🔴 Sidecars *can* carry execution state but lack canonical typing; implementations vary, and many tools will not preserve or validate these blobs — proven brittle in practice. | 🟢 GEN5 explicitly models latents + env as first-class chunks; enables replayability and causal experiments (the main scientific advantage). |
| Failure modes | 🟡 Sidecar failure mode = orphaning or silent loss (metadata disappears silently). Empirical reports show users losing sidecars in real workflows. | 🟢 GEN5 failure mode = unreadable/unsupported file (loud failure) — preferable for research because you detect lack of tooling. |
| Best practical use-cases | 🟢 Sidecars: photographers, mixed toolchains, read-only RAW editing, workflows that must avoid touching assets.| 🟢 GEN5: datasets, reproducible experiments, forensic archives, research pipelines where exact replayability matters (latent + env capture).|

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
    convert_float16=False, # whether to convert input tensors to float16
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
# now to save the metadata
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

# LICENSE
MIT
# Contribution
Please refer to the CONTRIBUTING.md filein the repo
# Documentation
Full docs: https://anuroopvj.github.io/gen5/api.html

# Future improvements
- Supporting other frameworks and utilities in the EnvChunk
- Reducing File Size

