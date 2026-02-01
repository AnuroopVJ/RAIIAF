MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10 gb default
MAX_CHUNK_SIZE = 2 * 1024 * 1024 * 1024  #2 gb per chunk
MAX_CHUNKS = 1000
# Header format:
#  - magic: 6s (6-byte ASCII "raiiaf")
#  - version_major: B (1 byte)
#  - version_minor: B (1 byte)
#  - flags: B (1 byte)
#  - chunk_table_offset: I (4 bytes)
#  - chunk_table_size: I (4 bytes)
#  - chunk_count: I (4 bytes)
#  - file_size: Q (8 bytes)
#  - reserved: I (4 bytes)
HEADER_FORMAT = "<6sBBBIIIQI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# JSON schema for the metadata chunk
JSON_SCHEMA = """{
"$schema": "http://json-schema.org/draft-07/schema#",
"title": "RAIIAF Metadata Schema",
"type": "object",

"properties": {
"raiiaf_metadata": {
    "type": "object",

    "properties": {

    "file_info": {
        "type": "object",
        "properties": {
        "magic":          { "type": "string", "const": "raiiaf" },
        "version_major":  { "type": "integer", "minimum": 1 },
        "version_minor":  { "type": "integer", "minimum": 0 },
        "file_size":      { "type": "integer", "minimum": 0 },
        "chunk_count":    { "type": "integer", "minimum": 0 }
        },
        "required": [
        "magic",
        "version_major",
        "version_minor",
        "file_size",
        "chunk_count"
        ]
    },

    "model_info": {
        "type": "object",
        "properties": {
        "model_name": { "type": "string" },
        "version":    { "type": "string" },
        "date":       { "type": "string" },
        "prompt":     { "type": "string" },
        "tags": {
            "type": "array",
            "items": { "type": "string" }
        },

        "generation_settings": {
            "type": "object",
            "properties": {
            "seed":         { "type": "integer", "minimum": 0 },
            "steps":        { "type": "integer", "minimum": 1 },
            "sampler":      { "type": "string" },
            "cfg_scale":    { "type": "number", "minimum": 0 },
            "scheduler":    { "type": "string" },
            "eta":          { "type": "number", "minimum": 0 },
            "guidance":     { "type": "string" },
            "precision":    { "type": "string" },
            "deterministic":{ "type": "boolean" }
            },
            "required": ["seed", "steps", "sampler"]
        },

        "hardware_info": {
            "type": "object",
            "properties": {
            "machine_name": { "type": "string" },
            "os":           { "type": "string" },
            "cpu":          { "type": "string" },
            "cpu_cores":    { "type": "integer", "minimum": 1 },

            "gpu": {
                "type": "array",
                "items": {
                "type": "object",
                "properties": {
                    "name":         { "type": "string" },
                    "memory_gb":    { "type": "number", "minimum": 0 },
                    "driver":       { "type": "string" },
                    "cuda_version": { "type": "string" }
                },
                "required": ["name"]
                }
            },

            "ram_gb":      { "type": "number", "minimum": 0 },
            "framework":   { "type": "string" },
            "compute_lib": { "type": "string" }
            },
            "required": ["os"]
        }
        },

        "required": [
        "model_name",
        "version",
        "date",
        "prompt",
        "tags"
        ]
    },

    "chunks": {
        "type": "array",
        "items": {
        "type": "object",
        "properties": {
            "index":             { "type": "integer", "minimum": 0 },
            "type":              { "type": "string" },
            "flags":             { "type": "string" },
            "offset":            { "type": "integer", "minimum": 0 },
            "compressed_size":   { "type": "integer", "minimum": 0 },
            "uncompressed_size": { "type": "integer", "minimum": 0 },
            "hash":              { "type": "string" },
            "extra":             { "type": "object" },
            "compressed":        { "type": "boolean" }
        },
        "required": [
            "index",
            "type",
            "flags",
            "offset",
            "compressed_size",
            "uncompressed_size",
            "hash",
            "extra",
            "compressed"
        ]
        }
    }
    },

    "required": ["file_info", "model_info", "chunks"]
}
},

"required": ["raiiaf_metadata"]
}"""
