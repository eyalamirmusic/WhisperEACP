# Plan

Derived from `CLAUDE.md`. The repository is set up, the toolchain is proven end
to end by `Tests/GPU`, the four modules the model is assembled from are in, and
the encoder is the layer being written out of them. This is the order those
layers come in, what "done" means for each, and what writing them surfaced in
the dependencies.

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

## The modules, landed

Each is a directory under `Lib/WhisperEACP/`, a target, and a test executable
under `Tests/`. The first four were written in parallel because none of them
includes a header from another — the seams below are the reason that holds.
`Encoder/` is the first module assembled on top of two of them, which is what
those seams were for.

150 tests across the suite. The ones that need a downloaded model or tokenizer
skip without one, the same shape as a GPU test returning early when
`Device::shared().isValid()` is false; configure with
`-DWHISPER_EACP_FETCH_MODEL=ON` and every one of them runs.

### `Kernels/` — `whisper-kernels`, `Tests/Kernels`

The op set a transformer is assembled from, each a `ComputeProgram` subclass in
its own header: layernorm, GELU, softmax, matmul.

These are the pieces the encoder and decoder are later written *out of*, so they
take shapes as uniforms rather than baking Whisper's in. A matmul that only
knows `tiny.en`'s widths is a matmul that gets rewritten for `base`.

The EDSL had `exp`, `log`, `sqrt`, `rsqrt`, `pow`, `max`, `min`, `clamp`, `mix`
and the rest of the usual set — but **no `tanh` and no `erf`**, which is exactly
the kind of gap this project exists to surface. GELU was first derived from what
was there, with the gap written down for eacp rather than hidden behind a
helper. eacp has both now (see the gaps section) and `Kernels/Gelu.h` calls its
`erf`. The switch was measured as a numerical no-op on Metal, digit for digit
(worst error 4.4e-7 against `std::erf` over [-4, 4], a thousand times inside
the tanh form whisper.cpp settles for), since eacp's helper is the same A&S
7.1.26 polynomial the kernel used to carry.

Softmax evaluates one `exp` per element: the exponential is parked in the
output buffer as it is summed and read back to be scaled in place, which is the
readable `OutputBuffer` at work. The emitted shader was checked to hold a single
`exp` call site, and a test pre-fills the output with a poison value so a read
that saw the pre-store value could not sum to one.

Since then the set has grown into what the encoder is assembled from, each in
its own header with a scalar reference in `Tests/Kernels`:

| kernel | what it is |
| --- | --- |
| `Conv1d` | PyTorch's `nn.Conv1d`, the weight read in its `[out, in, k]` file layout untransposed. A pair of output strides lets one kernel write channel-major for the next convolution or frame-major for the transformer — HF's permute after conv2, done by the store |
| `AttentionScores`, `AttentionApply` | the two halves of multi-head attention with `Softmax` between them. Query and key counts are separate uniforms, so cross-attention and a KV cache are the same kernels. The head is folded into the dispatch row, since eacp dispatches 1D or 2D and no further (see the gaps section) |
| `Add` | the residual and positional sums |
| `MatMulProgram<WeightStorage>` | `MatMul` as before, and `HalfWeightMatMul` reading the weight operand as packed fp16 through `readHalf`; identical uniforms, so a call site swaps only the alias |

The conv and attention kernels were mutation-checked rather than trusted green:
transposing a weight stride, dropping the padding guard's lower bound, or
swapping the head fold's divide and modulo each fails the tests.

### `Mel/` — `whisper-mel`, `Tests/Mel`

The front-end: 400-point STFT at a 160-sample hop, power spectrum, the 80 x 201
filterbank, `log10`, and Whisper's clamp-and-scale normalisation. 30 s in, an
80 x 3000 mel spectrogram out.

`log10` is eacp's intrinsic now. With the change of base gone, `log10(1e-10)`
comes back exactly where it used to miss by 1.1e-5, and the worst error over
the scalar-reference grid fell from 2.8e-6 to 2.3e-7.

The filterbank arrives as a buffer the caller binds, not as a table compiled in
— that is the seam that keeps this module independent of the loader that reads
`preprocessor_config.json`.

### `Model/` — `whisper-model`, `Tests/Model`

Safetensors: parse the JSON header with `<Miro/Json.h>` (which `eacp-core`
already brings), map each tensor's name to its dtype, shape and byte range in
the blob, and hand the bytes to a `GPU::Buffer`. `config.json` beside it.

`tiny.en` ships **F32**, not F16 — `"torch_dtype": "float32"`, and all 167
tensors in its `model.safetensors` are `F32`. F16 is what the larger repos
ship. `makeBuffer` sends F32 and F16 to the device as they lie in the blob —
the second as packed halves a kernel reads through `readHalf` — and returns a
`TensorBuffer` that names its storage, so a call site has to say which program
it binds rather than getting a bare buffer it can bind to the wrong one. BF16
and F64 still widen on the CPU, having no shader read of their own. An odd
count of halves is padded to a whole 32-bit word, because `readHalf` fetches
the word and would otherwise read past the allocation.

The file is memory-mapped through `eacp::MemoryMappedFile` rather than read
whole into a `Vector`: the three loads the tiny.en tests make peak at 63 MB
resident against the 151 MB file. `fromBytes` keeps the owned path for the
synthetic files the tests build, and both run through one `readHeader`, so
every validation is the same code on either.

The model itself is a download, never a commit. CPM fetches it, behind
`WHISPER_EACP_FETCH_MODEL` (default off), with `DOWNLOAD_NO_EXTRACT YES` and
`DOWNLOAD_ONLY YES` on each file.

### `Tokenizer/` — `whisper-tokenizer`, `Tests/Tokenizer`

BPE from `tokenizer.json`: vocab, merges, the byte-level pre-tokenizer, and
Whisper's special tokens. Encode and decode, with small committed fixtures for
the cases worth pinning.

### `Encoder/` — `whisper-encoder`, `Tests/Encoder`

HF's `WhisperEncoder.forward`, recorded into one command buffer out of
`Kernels/`, with the weights reaching it through `Model/`. `EncoderShape` is the
config plus a frame count — a parameter rather than 3000, so the real weights
can be run over a short input that a CPU reference can follow. `EncoderWeights`
loads every tensor by its HF name through `makeBuffer`, checks each shape
against the config, and names the tensor in a `ModelError` on a miss. `Encoder`
mirrors `MelSpectrogram`: `prepare()` compiles the nine programs and sizes every
intermediate once, `encode()` only records, one compute pass per dispatch
because a pass boundary is what orders one dispatch's writes against the next
one's reads.

`Linear` is the kernel it needed that `MatMul` was not: `y = x Wᵀ + b` with W in
`nn.Linear`'s `[out, in]` layout exactly as the file ships it, as
`LinearProgram<WeightStorage>` mirroring `MatMulProgram` so `HalfWeightLinear`
reads packed fp16. The encoder holds both and picks per projection by
`TensorBuffer::storage`; a packed tensor bound anywhere else — a convolution, a
layernorm, the positional sum — is a `ModelError` naming the tensor and the
kernel, since a packed buffer read through a float subscript is off by a factor
of two in every index, silently, on both backends.

Validation is the first tier throughout, against a double-precision encoder
written in the test from HF's definition rather than out of the kernels' own
references:

| test | tolerance | worst error measured |
| --- | --- | --- |
| small shape (6 mel bins, width 8, 2 heads, 2 layers, ff 16, 12 frames), float weights | 5e-6 | 3.3e-7 |
| the same with fp16 projections | 5e-6 | 2.5e-7 |
| tiny.en's real weights over 64 frames (32 positions, all 4 layers) | 1e-4 | 1.8e-5 |

in the mixed measure `|a - e| / (1 + |e|)`. The full 30 s window gives
1500 x 384 finite values in about 60 ms of GPU time on this machine — wall
clock around `commit()` from a Debug host build, which is the only clock
available (below) — from the naive per-thread kernels with no tiling.

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
| no `log10` | the same. `Mel/` calls it now; the change of base it replaced turned out to be where a 1.1e-5 error had been coming from (below) |
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

Downstream, the four workarounds these gaps had forced are gone: `Kernels/Gelu.h`
calls `erf`, `Mel/` calls `log10`, softmax evaluates one `exp` per element and
reads it back out of its `OutputBuffer`, and `Model/` keeps F16 packed with
`HalfWeightMatMul` reading it. Each was measured on the way out; the numbers
are in the module sections above.

One earlier note here was wrong and is corrected on the record: this file said
Metal's shader `log` was loose enough that `log10(1e-10)` came back as
-9.999989 (~14 ulp). It was the change of base, not the logarithm — the same
kernel on the same input returns exactly -10 through eacp's `log10`, and eacp's
own comment on the intrinsic predicted as much (an extra rounding on a value
that has already lost what a logarithm of a small number can lose). The Mel
tolerances were left where they were, because the same assertions are meant to
run against HLSL, where D3D specifies `log2` to 21 ulp, and a Metal measurement
cannot make that claim. eacp's own `log10` test allows ~1e-4 at |log| = 10 and
attributes it to Metal's logarithm; on this Mac the intrinsic is exact across
the twenty decades it sweeps, so that is a conservative bound reading as a
measurement.

### What assembling the encoder surfaced

The kernels above were written against eacp at `c13098b`, so this is the second
round of findings. None is worked around silently.

| finding | where, and what it means |
| --- | --- |
| dispatch is 1D or 2D only | `Frame/ComputePass.h:77` and `:82`, with `ThreadPosition` (`ShaderValue.h:513`) carrying only `x` and `y`. Attention is natively a (key, query, head) grid, and both backends take three counts — `MTLSize` has a depth, `ID3D12GraphicsCommandList::Dispatch` takes three — so this is the layer's rank limit rather than either backend's. The head is folded into the row on the host and unfolded with a divide and a modulo in every thread; `Attention.h` says so. The one place a real net wanted a shape the layer would not express |
| `Span::size()` is `int`; `MemoryMappedFile::bytes()` is not | `MemoryMappedFile.h:51` says the size is the full `size_t` "as is `bytes().getSize()`", but `Span::size()` asserts `fitsInt` (`ea_data_structures/Structures/Span.h:168`), so `mapped.bytes().size()` aborts in Debug and truncates in Release on exactly the file the class exists for. `Model/` uses `getSize()`; the fix upstream is a sentence beside `bytes()` |
| `MemoryMappedFile` is not move-assignable | `Pimpl` (`Core/Utils/Pimpl.h:17`) defaults its move constructor and declares no `operator=`, so `std::optional<MemoryMappedFile>` takes `emplace()` but rejects assignment, which is surprising for a type its header says to "move". A defaulted `Pimpl& operator=(Pimpl&&)` closes it |
| `readHalf` reads a whole word | `ShaderValue.h:2502` fetches `[index / 2]`, so a buffer holding an odd number of halves is read one word past its last whole word on the final element. Not written beside `readHalf`; the loader rounds the allocation up |
| no `graph()` on a shipped program | `ShaderProgram` exposes `source()` for the active backend but not its graph, so pinning the *other* backend's text for a real kernel means re-authoring its body on a bare `ShaderBuilder` — which is how eacp's own tests do it. An accessor would let a shipped kernel's HLSL be asserted on a Mac |
| loop-invariants are recomputed | the emitter gives up open names at a loop header by design (`StageEmitter`), so softmax's `1.0 / total` is divided once per element rather than hoisted. Harmless there; a real cost for a loop whose invariant is expensive and provably unwritten |
| `erf(0)` is 1e-9 | eacp's helper ends in `x < 0 ? -e : e` where the deleted local one ended in `sign(x) * (...)`, so it is not odd-symmetric at the origin. GELU multiplies it by `x == 0`, so nothing here sees it; a caller wanting `erf` itself would |
| `ComputePass` binds whole buffers only | `Frame/ComputePass.h:51` takes a `Buffer`, never the `BufferRange` that exists (`Buffer/Buffer.h:138`) and that `RenderPass::setVertexBuffer` (`Frame/RenderPass.h:145`) and `drawIndexed` already take. The encoder gets away with it because `embed_positions` is read as a prefix from offset 0; a KV cache, or a per-layer view of one packed tensor, has no way to bind at an offset short of a second buffer or an index uniform |
| no stated contract for aliasing an input and an output | nothing says whether a hoisted read survives a store through a differently declared pointer to the same allocation, so every 1:1 elementwise stage (`Gelu`, `Add`) writes to a separate buffer: 12 extra allocations, 108 MB of them the score and probability pair at the full window |
| GPU timing is `Frame`-only | `Timing/FrameTimer.h:15` is driven by `Frame`, and `CommandBuffer` has no timestamp hook, so an off-screen compute pass is timed by wall clock around `commit()` |
| no device-side zero fill | `Device::makeBuffer(int)` (`Device/Device.h:57`) allocates uninitialised, so k_proj's zero bias is a host vector uploaded once. Minor |

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
`eacp-core`, `bytes()` returning a span — and `Model/` now maps the weights
file through it. What that turned up is the `Span::size()` pairing in the table
above.

## After the encoder

The decoder, out of the same kernels: the token and positional embeddings are
row gathers, the self-attention runs `AttentionScores` and `AttentionApply`
with one query against the keys cached so far (which is why `queryCount` and
`keyCount` are separate uniforms), the cross-attention runs them against the
encoder's 1500 rows, and the logits are a `Linear` against the tied
`embed_tokens` matrix. What it needs that is not here yet: a gather kernel, an
argmax with Whisper's suppressed tokens masked, and the KV cache — which is
where `ComputePass` binding whole buffers only (above) bites first, since
appending a row means writing at an offset.

Then greedy sampling read back through `Tokenizer/`, whisper.cpp as the
stage-level oracle, and the streaming path through MakeASound. One thing to
know about the oracle before wiring it: whisper.cpp's public API takes a mel in
(`whisper_set_mel`) and gives logits out (`whisper_get_logits`), and does not
expose the encoder's output, so an encoder-level comparison against it goes
either through its internal state or through a decoder of ours. The front-end
and the tokenizer compare directly.
