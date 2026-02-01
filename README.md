# .raiiaf - Rich AI Image Artifact Format

![bannerimage](raiiaf_img.png)
[![Build and Deploy Docs](https://github.com/AnuroopVJ/RAIIAF/actions/workflows/docs.yml/badge.svg)](https://github.com/AnuroopVJ/RAIIAF/actions/workflows/docs.yml)
[![CI](https://github.com/AnuroopVJ/RAIIAF/actions/workflows/ci.yml/badge.svg)](https://github.com/AnuroopVJ/RAIIAF/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyPI](https://img.shields.io/pypi/v/raiiaf)

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

## Storage Efficiency Benchmark (Full Latent Tensor)

We evaluate the storage overhead of different industry-standard strategies for embedding large AI metadata by comparing how the same image and the same latent tensor are stored across multiple file formats.

### Experimental Setup
- **Images:** 5 PNG images
- **Latent tensor:** Shape (1, 4, 64, 64), approximately 89 KB
- **Metadata:** Identical semantic metadata across all formats
- **raiiaf implementation:** Official `raiiaf` API (v0.1.0), no mocks
- **Metric:** Relative file size overhead compared to the raw PNG baseline

### Compared Storage Strategies
- **Raw PNG:** Image only (baseline)
- **PNG + Embedded XMP:** Latent tensor serialized as XMP and embedded inside the PNG
- **PNG + XMP Sidecar:** Latent tensor stored in a separate `.xmp` file alongside the PNG
- **raiiaf (.raiiaf):** Single-file, binary container storing both image and latent tensor

### Results
The average file sizes and relative overheads are summarized below:

| Format | Avg. Size (KB) | Avg. Overhead (%) |
|------|---------------|-------------------|
| Raw PNG | 1708.3 | – |
| RAIIAF (.raiiaf) | 1739.3 | 1.8 |
| PNG + Embedded XMP | 1797.9 | 5.2 |
| PNG + XMP Sidecar | 1795.2 | 5.1 |

![Storage overhead comparison for different metadata strategies](paper/graph.png)

### Interpretation
For the same image and identical latent tensor, XMP-based workflows incur approximately 5.1–5.2% storage overhead, regardless of whether the metadata is embedded or stored as a sidecar file. In contrast, raiiaf introduces only ~1.8% overhead.

This corresponds to a ~3.3% absolute reduction in file size and approximately 65% lower relative metadata overhead compared to standard XMP-based approaches, while preserving a single-file workflow.

These results indicate that a binary, AI-native container can store large latent tensors more space-efficiently than XML-based metadata strategies under identical conditions.

# Why not just use EXIF/XMP/Sidecar files?
Here is an emperical comparison:

| Aspect | EXIF / XMP (Custom Metadata) | raiiaf |
|------|-------------------------------|------|
| Schema enforcement | Convention-based, unenforced | 🟢 Canonical, versioned schema |
| Semantic consistency | Low; tag drift common | 🟢 High; fixed fields + chunk types |
| Latent representation | Not supported (text hacks) | 🟢 Native latent chunks (binary-safe) |
| Environment capture | Ad-hoc text notes | 🟢 Explicit env chunks (model, seed, hw, libs) |
| Reproducibility ceiling | Limited, state incomplete | 🟢 High; full generation state captured |
| Data typing | Weak (string-heavy XML) | 🟢 Strong typing (binary, arrays, structs) |
| Extensibility | Easy but uncontrolled | 🟡 Controlled; safer but slower evolution |
| Tooling ecosystem | Mature, ubiquitous | 🔴 Immature, raiiaf-specific tools needed |
| Interoperability | Works almost everywhere | 🔴 Breaks without raiiaf-aware readers |
| Failure mode | Metadata silently ignored | 🟡 Metadata explicit but unreadable without tooling |

Although XMP has the ability to embed any binary payload, doing so necessitates ad hoc conventions, manual validation, and cautious handling to prevent silent data loss. By formalizing these practices into a first-class, schema-enforced representation, raiiaf lessens failure modes and implementation burden.

a comparison with sidecar files: 

| Aspect | Where Sidecar (XMP) shines / empirical behavior | Where raiiaf shines / empirical behavior |
|---|---|---|
| Non-destructive editing | 🟢 Sidecars allow metadata edits without touching the original asset (important for read-only/ RAW workflows). Many DAMs expose "write to sidecar only" modes. | 🟡 raiiaf typically embeds state into a container; editing metadata might require raiiaf-aware tooling. Good for integrity but less “drop-in” non-destructive edit unless you design a sidecar-style raiiaf wrapper. |
| Support & tooling (read/write) | 🟢 Very broad: exiftool, Lightroom, digikam and many tools understand XMP and sidecars; exiftool can create and sync sidecars. This is production proven.| 🔴 Immature: raiiaf requires custom readers/writers; integration work required. This is the main adoption barrier (engineering cost). |
| Round-trip fidelity (read->modify->write->read) | 🔴 Variable: many apps will normalize/sync/overwrite XMP fields; different tools may not round-trip custom blobs reliably (risk of normalization or loss). Empirical reports of inconsistent XMP syncing and sidecar issues exist. | 🟢 High if tools are spec-compliant: raiiaf’s schema + chunk validation lets compliant tools preserve unknown chunks and guarantee round-trip fidelity (design goal). |
| Orphaning & file management | 🔴 Sidecars can be orphaned (moved, renamed, or not uploaded); cloud/backup processes often miss them — real world bug reports and GH issues. | 🟢 raiiaf embeds state into the artifact (single file), eliminating the orphaning problem — better for long-term datasets and archives. |
| Resilience to platform stripping | 🔴 Both can be stripped in public pipelines; embedded XMP is sometimes removed by social platforms and services. Sidecars are even more fragile because many uploaders ignore sidecar files. | 🟡 raiiaf helps when you control the pipeline (archives, datasets). For public publishing (social platforms) nothing is guaranteed unless the platform preserves custom blocks — but embedding reduces the chance of separate-file loss. |
| Validation & semantics | 🔴 XMP allows arbitrary namespaces; no enforcement — different users/tools will store semantically identical things under different keys (fragmentation). Empirical evidence of inconsistent metadata across tools.| 🟢 raiiaf enforces typed schema, versioning, and chunk semantics → machine-verifiable metadata (reduces semantic drift). This is the core reproducibility gain. |
| Indexing & large-scale querying | 🟡 Sidecars can be indexed when present, but inconsistent field names and scattered sidecars complicate large dataset indexing.| 🟢 raiiaf’s structured schema makes programmatic indexing and deterministic query semantics straightforward — ideal for datasets and benchmarks. |
| Forensic reproducibility (latents + env) | 🔴 Sidecars *can* carry execution state but lack canonical typing; implementations vary, and many tools will not preserve or validate these blobs — proven brittle in practice. | 🟢 raiiaf explicitly models latents + env as first-class chunks; enables replayability and causal experiments (the main scientific advantage). |
| Failure modes | 🟡 Sidecar failure mode = orphaning or silent loss (metadata disappears silently). Empirical reports show users losing sidecars in real workflows. | 🟢 raiiaf failure mode = unreadable/unsupported file (loud failure) — preferable for research because you detect lack of tooling. |
| Best practical use-cases | 🟢 Sidecars: photographers, mixed toolchains, read-only RAW editing, workflows that must avoid touching assets.| 🟢 raiiaf: datasets, reproducible experiments, forensic archives, research pipelines where exact replayability matters (latent + env capture).|

## Installation
Just pip install the package!
```bash
pip install raiiaf
```
## Usage
import the classes
```python
from raiiaf.main import raiiafFileHandler
```
First you need to instantiate the raiiafFileHandler class.
```python
raiiaf = raiiafFileHandler()
```

# Encoding
!!! danger
    **DISCLAIMER**:
    The encoder expects **NumPy arrays**.  
    If you use PyTorch tensors, convert them with `.detach().cpu().numpy()`.

```python
from raiiaf.main import raiiafFileHandler

raiiaf = raiiafFileHandler()
initial_noise_tensor = torch.randn(batch_size, channels, height, width)
latent = {
    "initial_noise": initial_noise_tensor.detach().cpu().numpy() #The encoder expects numpy array not a torch tensor object
}
binary_img_data = raiiaf.png_to_bytes(r'path/to/image.png') # use the helper function to convert image to bytes

raiiaf.file_encoder(
    filename="encoded_img.raiiaf", # The .raiiaf extension is required!
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
decoded = raiiaf.file_decoder(filename)
# now to save the metadata
metadata = decoded["metadata"]["raiiaf_metadata"]

# to just get specific metadata blocks
model_info = decoded["metadata"]["raiiaf_metadata"]["model_info"]

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
Full docs: https://anuroopvj.github.io/RAIIAF

# Future improvements
- Supporting other frameworks and utilities in the EnvChunk
- Reducing File Size

