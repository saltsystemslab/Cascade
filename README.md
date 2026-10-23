# Cascade

**GPU-accelerated out-of-core suffix array construction for LLM corpora.**

[![arXiv](https://img.shields.io/badge/arXiv-2610.15254-b31b1b.svg)](https://arxiv.org/abs/2610.15254)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Suffix arrays power deduplication, contamination testing, n-gram statistics, and training-data attribution over the corpora used to train large language models, and every new corpus or tokenizer needs a new one.
Cascade builds the suffix array of a tokenized corpus on a single GPU server, for corpora many times larger than the GPU's memory, and searches it with a batched query engine.

While LLM corpora motivate its design, nothing in it depends on them.
Cascade can index any sequence of symbols, such as genome collections or plain bytes.

<p align="center">
  <img src="docs/assets/build-throughput.png" width="720"
       alt="Build throughput per corpus, Cascade against infini-gram and libsais">
</p>

> **Status: research release.** This is the code evaluated in the [paper](https://arxiv.org/abs/2610.15254).
> A packaged release is in progress: one `cascade` command with flags instead of environment variables, builds for more GPU generations, documentation, and Python bindings.
> The interface below is subject to change.

## Why Cascade?

- **Fast.** Cascade builds the suffix array 3.0–8.2× faster end to end than [infini-gram](https://infini-gram.io/)'s indexing pipeline on seven corpora of 8–238 billion tokens, and its construction is 6.3–11.4× faster than [libsais](https://github.com/IlyaGrebnov/libsais)'s.
- **Not bound by GPU memory.** Cascade sorts its partitions on the GPU independently and never merges them.
  Thus GPU memory limits only the partition size and not the corpus size.
  With just 5 GB of GPU memory (a sixteenth of the peak usage), suffix array builds over the arxiv (30B) and the github (64B) corpora take only 1.3× and 1.5× as long as the builds that use the full GPU memory.
- **Light on host memory.** Cascade streams the ranks of suffixes between partitions as messages, instead of storing them in a corpus-wide rank table (the inverse suffix array, ISA).
  Its intermediate data gracefully spills across GPU memory, host memory, and SSD.
  Host memory only needs to hold the text: on our server with 1 TB of host DRAM, Cascade indexes the 198B-token c4 and the 238B-token pile corpora in 23 and 37 minutes, where libsais runs out of memory.
- **Exact.** Cascade never truncates suffix comparisons, and thus the suffix arrays are exact.
- **Fast queries.** Cascade's batched query engine reproduces infini-gram's counts at median speedups of 10.8–32.6×, with the index in GPU memory, host memory, or on SSD.

## How it works

Cascade samples K-grams of the text to choose splitters that give each partition a fixed, contiguous range of the final suffix array.
The GPU then sorts each partition, iteratively in generations, by prefix doubling.
Between generations, every rank is sent directly to the one partition that needs it, as a message.
An invariant keeps every rank comparison within the same partition, so partition-local ranks suffice.
These messages and the suffixes' records are elastic: they fill the room left in host memory and spill to SSD beyond it.
Each finished partition is written directly to its range of the output.
The [paper](https://arxiv.org/abs/2610.15254) has the details and the proofs.

## Requirements

- Linux on x86-64 (the host code uses AVX2, `O_DIRECT`, and `fallocate`).
- An NVIDIA GPU.
  The default build targets compute capability 12.0, such as the RTX PRO 6000 Blackwell we evaluate on.
  Set `CUDA_ARCH` for other GPUs (see below).
- CUDA 12.8, which bundles CCCL 2.7, and GCC 11: the versions we build with.
- CMake 3.22 or newer and OpenMP.
- Host memory for the text (2 bytes per token) plus working buffers.
- Node-local NVMe drives for the spills, on a filesystem supporting `O_DIRECT` and `fallocate`, such as XFS or ext4.
  The output takes 5 bytes per token.
  Our three largest corpora (github, c4, and pile) spill 6–20 bytes per token.

## Building

On the machine with the GPU:

```bash
cmake -S . -B build -DCUDA_ARCH=120-real
cmake --build build -j
```

`CUDA_ARCH` takes CMake's architecture syntax: e.g. `80-real` for an A100, `90-real` for an H100 or H200.
The build produces `build/src/cascade` (the builder), `build/src/query` (the query engine), and `build/tools/sa_convert`.

`./build.sh` only compile-checks the same build in a CUDA 12.8 Docker image, without a GPU, writing to `./build`.
Its binaries are linked against that image's Ubuntu 22.04 libraries, so build Cascade natively on the machine that runs its binaries.

## Preparing the input

The builder reads a raw array of little-endian 16-bit symbols.
Neither the construction nor the suffix array assumes anything about the symbols beyond their order.
Cascade works with any sequence, such as the token IDs of an LLM corpus or the bytes of a text or a genome.
8- and 32-bit inputs are in the roadmap; until then, widening a byte text to 16 bits leaves its suffix array unchanged, at twice the text's memory:

```bash
python3 -c "import numpy as np; np.fromfile('genome.seq', np.uint8).astype('<u2').tofile('genome.u16')"
```

For LLM corpora, our evaluation uses infini-gram's `tokenized.N` layout, with the separator `0xFFFF` before each document.
`dataset/tokenize-corpus.py` writes one from JSONL or Parquet files with any Hugging Face tokenizer of at most 65,535 tokens:

```bash
python3 dataset/tokenize-corpus.py --input-dir raw/ --output corpus.tokenized.bin --tokenizer gpt2
```

It reads `*.jsonl*` files by default (plain, `.gz`, or `.zst`); pass `--glob '*.parquet'` for Parquet.
It needs `transformers`, `tokenizers` and `numpy`, plus `pyarrow` for Parquet, `zstandard` for `.zst` files, and `sentencepiece` for Llama-2's tokenizer.
`dataset/` also contains scripts that download, tokenize, and verify the corpora of the paper.

## Building a suffix array

```bash
GPU_SACA_SCRATCH=/nvme0/scratch:/nvme1/scratch build/src/cascade corpus.tokenized.bin corpus.sa
```

- `GPU_SACA_SCRATCH` (required): colon-separated directories on node-local drives for the spills.
  Cascade stripes across directories if multiple are provided (e.g. one per drive).
- `GPU_SACA_RAM_BUDGET` (optional): the host-memory budget in bytes; the default is 90% of the available memory.
  It doesn't observe cgroup limits yet, so set it inside containers and Slurm jobs that limit memory.
- `OMP_NUM_THREADS`: number of CPU threads to use.

The output holds one 5-byte little-endian entry per token: the starting positions of the suffixes, in lexicographic order of the token IDs.

## Querying

Cascade's query engine counts the occurrences of batches of token sequences.
Wrap the suffix array into an index, generate a batch of queries, and run it:

```bash
build/tools/sa_convert --meta-only --sa corpus.sa --text corpus.tokenized.bin --sym-bytes 2 --index-dir corpus.idx
build/src/query gen --index corpus.idx --out q.qry --kind substring --q 100000 --len 4,8,16
build/src/query find --index corpus.idx --queries q.qry --text-res auto --sa-res auto --stride auto \
    --reps 1 --cnt-out q.cnt
```

`q.cnt` holds one little-endian `uint64` count per query.
The engine places the index in GPU memory, host memory, or on SSD, whichever fits.
`corpus.idx/index.meta` records where the text and the suffix array are, so leave them in place.
Running `build/src/query` without arguments prints the full usage, and `include/query-file.hpp` describes the `.qry`, `.itv`, and `.cnt` formats.

## Citation

If you use Cascade, please cite:

```bibtex
@misc{khan2026cascade,
  title         = {Cascade: {GPU}-Accelerated Out-of-Core Suffix Array Construction for {LLM} Corpora},
  author        = {Khan, Jamshed and Pandey, Prashant},
  year          = {2026},
  eprint        = {2610.15254},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DS},
  doi           = {10.48550/arXiv.2610.15254},
}
```

## License

Cascade is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for the third-party components.
