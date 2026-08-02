# DeepSeek-V4-Flash CPU support

Colibri supports the `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint through a
separate, CPU-only CLI engine.  The runtime uses no Python or third-party
library; PyTorch and safetensors are needed only while converting the official
Hugging Face snapshot.

## Convert the official snapshot

Download the snapshot to an SSD, then convert it shard by shard:

```sh
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /models/DeepSeek-V4-Flash-0731
python c/tools/convert_deepseek_v4.py \
  --indir /models/DeepSeek-V4-Flash-0731 \
  --outdir /nvme/DeepSeek-V4-Flash-0731-colibri \
  --max-gb 170
```

The converter keeps routed experts in the official MXFP4 E2M1 + UE8M0/g32
layout, converts non-expert matrices to grouped int4, and writes a manifest.
`--include-dspark` stores the optional MTP/DSpark tensors in `dspark/`; the
base CLI does not load that sidecar.

The ratio-4 learned indexer tensors are retained in the converted snapshot for
future use, but the first 15-GB CPU engine selects those blocks with a
deterministic main-query top-k approximation rather than loading a second
indexer/compressor cache.  Use the reference-output comparison below to
measure the quality impact on the target checkpoint.

## Build and run

```sh
make -C c deepseek_v4 ARCH=x86-64-v3
COLI_MODEL=/nvme/DeepSeek-V4-Flash-0731-colibri \
  python c/coli run --ngen 64 "Explain the difference between RAM and VRAM."
```

For a 15 GB x86-64 AVX2 laptop, keep the model directory on NVMe and use a
50,000-token context.  The engine reads selected routed experts from SSD and
releases them after each token; it does not attempt to make the 160–170 GB
model resident in RAM.  `CTX`, `NGEN`, and `PROMPT` are also accepted directly
by `deepseek_v4` for low-level testing.

The initial engine is intentionally CLI-only and greedy by default so fixed
reference prompts produce repeatable outputs.  Record cold-SSD throughput with
the OS page cache dropped between runs, and report CPU model, AVX2 status,
NVMe model, context length, and the converted manifest alongside the tok/s
measurement.

The repository includes a safe expert-cache-cold runner.  It uses
`posix_fadvise(DONTNEED)` after each streamed expert read instead of requiring a
privileged system-wide cache flush:

```sh
python c/tools/bench_deepseek_v4.py \
  --engine c/deepseek_v4 \
  --model /nvme/DeepSeek-V4-Flash-0731-colibri \
  --prompt '<｜User｜>Give a one-sentence answer.<｜Assistant｜>' \
  --tokens 32 --ctx 50000 --repeats 3 --cold-expert
```

Pass `--reference fixed-output.txt` to the same command to emit exact-match
and `difflib` sequence-similarity measurements for the fixed quality replay.

The tensor names and FP4 preservation follow the official
[DeepSeek-V4 inference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py)
and [official converter](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/convert.py).
