# Plan

Derived from `CLAUDE.md`. The repository is set up, the toolchain is proven end
to end by `Tests/GPU`, and every model layer above `Audio/` is unwritten. This
is the order those layers get written in, and what "done" means for each.

## Where the ground truth lives

| question | answer |
| --- | --- |
| audio contract | `Lib/WhisperEACP/Audio/Format.h`, pinned by value in `Tests/Audio` |
| model shapes | `config.json` from the HF repo |
| mel filterbank | `preprocessor_config.json`, an 80 x 201 matrix, already built |
| weights | `model.safetensors` — JSON header, then a raw blob |
| vocabulary | `tokenizer.json` — BPE vocab and merges |

Nothing here reimplements librosa's filterbank construction or lifts one out of
another project: the matrix is in the file.

## The four modules, landed

Each is a directory under `Lib/WhisperEACP/`, a target, and a test executable
under `Tests/`. They were written in parallel because none of them includes a
header from another — the seams below are the reason that holds, and they still
hold now that all four are in.

96 tests across the four, on top of the 11 that were here before. 16 of them
need a downloaded model or tokenizer and skip without one, the same shape as a
GPU test returning early when `Device::shared().isValid()` is false.

### `Kernels/` — `whisper-kernels`, `Tests/Kernels`

The op set a transformer is assembled from, each a `ComputeProgram` subclass in
its own header: layernorm, GELU, softmax, matmul.

These are the pieces the encoder and decoder are later written *out of*, so they
take shapes as uniforms rather than baking Whisper's in. A matmul that only
knows `tiny.en`'s widths is a matmul that gets rewritten for `base`.

The EDSL has `exp`, `log`, `sqrt`, `rsqrt`, `pow`, `max`, `min`, `clamp`, `mix`
and the rest of the usual set — but **no `tanh` and no `erf`**, which is exactly
the kind of gap this project exists to surface. GELU is derived from what is
there, and the gap is written down for eacp rather than hidden behind a helper.

### `Mel/` — `whisper-mel`, `Tests/Mel`

The front-end: 400-point STFT at a 160-sample hop, power spectrum, the 80 x 201
filterbank, `log10`, and Whisper's clamp-and-scale normalisation. 30 s in, an
80 x 3000 mel spectrogram out.

The filterbank arrives as a buffer the caller binds, not as a table compiled in
— that is the seam that keeps this module independent of the loader that reads
`preprocessor_config.json`.

### `Model/` — `whisper-model`, `Tests/Model`

Safetensors: parse the JSON header with `<Miro/Json.h>` (which `eacp-core`
already brings), map each tensor's name to its dtype, shape and byte range in
the blob, and hand the bytes to a `GPU::Buffer`. `config.json` beside it.

`tiny.en` ships **F32**, not F16 — `"torch_dtype": "float32"`, and all 167
tensors in its `model.safetensors` are `F32`. F16 is what the larger repos
ship, so the loader handles both and widens on the way in: eacp's EDSL has no
half type at all, so packed halves cannot be read by a kernel and the
conversion would only move into every kernel that touches a weight.

The model itself is a download, never a commit. CPM fetches it, behind
`WHISPER_EACP_FETCH_MODEL` (default off), with `DOWNLOAD_NO_EXTRACT YES` and
`DOWNLOAD_ONLY YES` on each file.

### `Tokenizer/` — `whisper-tokenizer`, `Tests/Tokenizer`

BPE from `tokenizer.json`: vocab, merges, the byte-level pre-tokenizer, and
Whisper's special tokens. Encode and decode, with small committed fixtures for
the cases worth pinning.

## Validation, in the order a failure is diagnosed

1. **A scalar CPU reference inside the test itself**, for every kernel. No
   dependency, runs anywhere, and the only tier that catches a backend
   divergence — the same assertion runs against MSL here and HLSL on Windows.
   This is the bulk of the suite.
2. **whisper.cpp as an in-process oracle**, not yet wired up. Stage-level, so a
   mismatch bisects to a layer instead of reporting a wrong transcript.
3. **Small committed fixtures** for what is cheap and stable.

A GPU test returns early when `Device::shared().isValid()` is false, so the
suite still passes on a machine with no GPU.

## Gaps found in eacp and Miro

The second goal, in CLAUDE.md's terms: writing a real net against eacp's compute
layer is what surfaces what that layer is missing, and a gap belongs there
rather than in a workaround here. Each of these was confirmed against the
dependency's own source, not inferred from a compile error. None is worked
around silently — the substitute is named at the point it is used.

### The shader EDSL

| gap | evidence | what it costs here |
| --- | --- | --- |
| no `tanh`, `sinh`, `cosh` | absent from the emitter's intrinsic table | native in **both** MSL and HLSL — pure EDSL surface |
| no `log10` | same | `Mel/` changes base by `0.43429448190325176f` |
| no `erf` / `erfc` | same | native in MSL, **absent from HLSL** — this one needs a polynomial fallback in the emitter, not just a name. `Kernels/Gelu.h`'s A&S 7.1.26 is the raw material: 4.4e-7 max error against exact GELU, where the tanh approximation whisper.cpp uses costs 4.7e-4 |
| `InputBuffer::operator[]` takes only `const UInt&` | `AtomicBuffer::load(unsigned)` and `Shared<T>::operator[](unsigned)` both have literal overloads, and the comment on the first calls it "the same courtesy the intrinsics extend to a float literal" | reading element 0 of a one-element buffer needs a `var(0u)` to manufacture an index |
| no half type | `ValueType` is `Float/Float2/.../UInt/Int/Bool`; `InputBuffer` yields `Float` | why `Model/` widens F16 on load. A large-v3 in F16 pays 2x GPU memory for it |
| `Uniform<InputBuffer>` stores a pointer | assigning a temporary compiles and dangles, with no diagnostic | every bound buffer must be a named local |
| `OutputBuffer` is write-only | no read accessor | softmax evaluates `exp` twice per element |
| `GPU::Buffer` sizes are `int` | `makeBuffer(const void*, int)` | caps one buffer at 2 GB |
| README drift | documents `sharedArray<T, N>()` and `threadIndexInGroup()` | the header has `shared<T>(int)` and `localId()` |

Not a gap, and worth knowing: Metal's shader `log` is loose enough that
`log10(1e-10)` comes back as -9.999989 (~14 ulp), so a test on the log of a
small quantity needs ~1e-4 rather than 1e-6.

### Miro

| gap | evidence |
| --- | --- |
| `Json` mis-decodes surrogate pairs | `parseUnicodeEscape` reads four hex digits with no pairing, and `appendUtf8` has no 4-byte branch. `"😀"` parses to `ED A0 BD ED B8 80` — CESU-8 — instead of `F0 9F 98 80`. Any `tokenizer.json` written with Python's default `ensure_ascii=True` decodes astral characters wrongly; the HF file is raw UTF-8, which is the only reason it does not bite |
| `Json::Value::operator[]` uses `std::map::at` | a missing key throws `std::out_of_range` rather than a JSON error, so `Miro::Json::find` is the only safe accessor |
| `Json::Object` is `std::map<std::string, Value>` | 152 ms to parse `tokenizer.json`'s 50k-key vocabulary in Debug. The safetensors header has the same shape |
| `Json` numbers are all `double` | no integer preservation, so 64-bit safetensors offsets need an explicit whole-number and 2^53 range check |
| no Unicode general-category API | the GPT-2 pre-tokenizer's `\p{L}` / `\p{N}` classes forced a generated 1585-entry table into `Tokenizer/Unicode.cpp`. Text layout or IME handling would need the same data |

### Checked and discarded

`eacp::MemoryMappedFile` **does** exist — commit `3081829`, compiled into
`eacp-core`, `bytes()` returning a span. `Model/` reads the 151 MB blob with an
`ifstream` instead, so the whole file is resident before the GPU copy. That is
ours to fix, not eacp's.

## After these four

The encoder (conv front-end, then the transformer blocks), the decoder with its
cross-attention and KV cache, and greedy sampling — assembled from `Kernels/`,
fed by `Mel/` and `Model/`, read back through `Tokenizer/`. Then whisper.cpp as
the stage-level oracle, and the streaming path through MakeASound.
