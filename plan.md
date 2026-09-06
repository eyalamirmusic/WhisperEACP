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

The EDSL had `exp`, `log`, `sqrt`, `rsqrt`, `pow`, `max`, `min`, `clamp`, `mix`
and the rest of the usual set — but **no `tanh` and no `erf`**, which is exactly
the kind of gap this project exists to surface. GELU was derived from what was
there, with the gap written down for eacp rather than hidden behind a helper.
eacp has both now (see the gaps section), and `Kernels/Gelu.h` still carries
its own polynomial until it is switched over.

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
ship, so the loader handles both and widens on the way in. That was forced:
eacp's EDSL had no way to read a packed half. It has one now — `readHalf`, in
the gaps section — so keeping F16 packed on the GPU is the loader's next
change rather than a constraint.

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

All of these but one are fixed upstream, in eacp `develop` from commit `c13098b`,
which this project fetches. Each was proven by a failing test in eacp's own
`Tests/GPU` before the fix, and the emitted MSL and HLSL are both asserted as
text on the Mac, since the emitter is pure string generation.

| gap | what eacp does now |
| --- | --- |
| no `tanh`, `sinh`, `cosh` | componentwise intrinsics, native in both languages |
| no `log10` | the same. `Mel/` still changes base by `0.43429448190325176f` until it is switched over |
| no `erf` / `erfc` | this was recorded as native in MSL, and that was wrong: MSL has **no `erf` either** — the Metal compiler rejects it as an undeclared identifier. So the A&S 7.1.26 polynomial is the definition on *both* backends, emitted as a helper with `float2/3/4` overloads. Under 6e-7 absolute against `std::erf` on the CPU, 1.7e-7 on Metal. `erfc` is its own form rather than `1 - erf`, so the tail keeps its digits; its relative error out there is still the approximation's, around 1 percent by x = 3 |
| `InputBuffer::operator[]` takes only `const UInt&` | an `unsigned` overload on the subscript and on `read2`/`read3`/`read4`, anchored on the buffer's graph the way `AtomicBuffer::load(unsigned)` is |
| no half type | there is deliberately still none: the Windows backend compiles through FXC at `cs_5_0`, where `half` is a synonym for `float`, so a `Half` would mean two different things on the two backends. What landed is fp16 **storage** with fp32 arithmetic: `InputBuffer::readHalf(i)` (element `i` of a buffer of halves, widened), `readHalf2(i)`, `packHalf2`, `asFloat` and `writeHalf2`. Widening is exact; narrowing is the one place the backends differ — Metal rounds to nearest-even and overflows to infinity, D3D specifies round-toward-zero saturating to 65504 |
| `Uniform<InputBuffer>` stores a pointer | every `Uniform<resource>` — the three buffers and the four textures — deletes its rvalue `operator=`, so assigning a temporary is a compile error |
| `OutputBuffer` is write-only | readable through the same subscript and `read2`/`read3`/`read4`; both backends already declared it writable, so nothing new is bound. What it promises is read-after-write within one thread. Fixing it exposed an emitter bug: a buffer read hoisted to a local was never retired by a store to the same slot, so a reused read saw the pre-store value. Stores and atomic adds now stale such names, the rule variables and shared memory already had |
| README drift | corrected to `shared<T>(count)` and `localId()` |

The one left alone is **`GPU::Buffer` sizes are `int`**. eacp `6ef2cde` moved
every interface to `int` on purpose the day before `c13098b`, keeping `size_t`
only where the language or an OS API forces it, so widening `Buffer` is a
decision for the maintainer rather than a gap to fill. No single Whisper tensor
approaches 2 GB.

Nothing here ran on Windows: the HLSL side is asserted as emitted text only,
and D3D's `f32tof16` rounding is from the D3D11.3 functional spec rather than a
run.

Downstream, the workarounds these gaps forced are still in place, and taking
them out is what comes next: `erf()` in `Kernels/Gelu.h`, `log10` in `Mel/`,
one `exp` per element in softmax, and `Model/` keeping F16 packed instead of
widening.

Not a gap, and worth knowing: Metal's shader `log` is loose enough that
`log10(1e-10)` comes back as -9.999989 (~14 ulp), so a test on the log of a
small quantity needs ~1e-4 rather than 1e-6.

### Miro

All five gaps below are fixed upstream, on Miro `main` from commit `948b34e`, which
eacp fetches. Each was proven by a failing test in Miro's own suite before the fix.

| gap | what Miro does now |
| --- | --- |
| `Json` mis-decoded surrogate pairs | `parseUnicodeEscape` pairs `\uD8xx\uDCxx` into one code point and emits 4-byte UTF-8. A lone or mismatched surrogate, or a non-hex digit in a `\u` escape, is a `ParseError` |
| `Json::Value::operator[]` threw `std::out_of_range` | `Miro::Json::Error` is the base of `ParseError` and the new `AccessError`, which the `as*()` accessors and implicit conversions throw on a type mismatch, naming both types. `operator[]` itself is deliberately left unchecked, by decision: a missing key or bad index stays the caller's error, and `find` remains the accessor for an optional key |
| `Json::Object` cost 152 ms for a 50k-key parse in Debug | Measured, not assumed. 83 to 90 percent of that parse is inside `std::map::emplace`, in key comparison during the tree descent rather than allocation, so nothing inside the parser can move it. `Object` gained a transparent comparator so `find` takes a `string_view` without allocating (Debug lookups 19 percent faster). Swapping the container would parse 2.4 to 2.6x faster and look up 5 to 6x faster, at the cost of sorted iteration order in `print()`. Left as the maintainer's call |
| `Json` numbers were all `double` | An `std::int64_t` alternative. A literal with no fraction or exponent parses exactly, `isInteger()` / `asInteger()` expose it, `isNumber()` / `asNumber()` still answer for both kinds, equality is numeric across them, and the reflection layer carries `int64` through typed and raw fields. Doubles print shortest-round-trip instead of six significant digits |
| no Unicode general-category API | `<Miro/Unicode.h>`: all 30 general categories, the `\p{..}` class predicates, the `White_Space` property, and strict UTF-8 `decodeUtf8` / `appendUtf8`, from a table generated out of Unicode 16.0 and checked against Python's `unicodedata` for every code point |

Two things surfaced on the way that belong on the record. Miro's first push used the
floating-point `std::to_chars`, which libc++ marks unavailable below macOS 13.3; this
project targets 11.0, so the fetch failed to compile until `948b34e` printed doubles
through a shortest-round-trip `snprintf` loop instead. And `EA::Vector::operator[]`
is unchecked and `noexcept`, so an out-of-range JSON array index was undefined
behaviour before and still is.

Downstream, the two workarounds these gaps had forced here are gone: `Model/`'s
integer reads go through `asInteger()` instead of a 2^53 range check on a double, and
`Tokenizer/Unicode.cpp` is a thin three-class view over `Miro::Unicode` instead of a
private 1585-entry table.

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
