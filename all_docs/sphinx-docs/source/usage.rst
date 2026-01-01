Usage
=====

.. code-block:: python

    from gen5.file_encoder import file_encoder
    from gen5.file_decoder import file_decoder
    import json
    from PIL import Image
    import io

    gen5.file_encoder(
        filename="encoded_img.gen5",  # The .gen5 extension is required!
        latent=latent,  # initial latent noise
        chunk_records=[],
        model_name="Stable Diffusion 3",
        model_version="3",  # Model Version
        prompt="A puppy smiling, cinematic",
        tags=["puppy", "dog", "smile"],
        img_binary=binary_img_data,
        convert_float16=False,  # whether to convert input tensors to float16
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
            "cpu_cores": 8,  # minimum 1
            "gpu": [{"name": "RTX 3090", "memory_gb": 24, "driver": "nvidia", "cuda_version": "12.1"}],
            "ram_gb": 64.0,
            "framework": "torch",
            "compute_lib": "cuda"
        }
    )

    # Decoding
    decoded = gen5.file_decoder("encoded_img.gen5")
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