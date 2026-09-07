# Plan

Derived from `CLAUDE.md`. The repository is set up, the toolchain is proven end
to end by `Tests/GPU`, the four modules the model is assembled from are in, and
the encoder and decoder are the layers written out of them. This is the order
those layers come in, what "done" means for each, and what writing them
surfaced in the dependencies.

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
`Encoder/` and `Decoder/` are the two modules assembled on top of them, which is
what those seams were for — neither includes the other's header, since the
encoder's output reaches the decoder as a buffer. `Whisper/` is the call site
that owns both and turns 30 seconds of audio into a string.

222 tests across the suite, with the oracle on. The ones that need a downloaded model or tokenizer
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

Since then the set has grown into what the encoder and decoder are assembled
from, each in its own header with a scalar reference in `Tests/Kernels`:

| kernel | what it is |
| --- | --- |
| `Conv1d` | PyTorch's `nn.Conv1d`, the weight read in its `[out, in, k]` file layout untransposed. A pair of output strides lets one kernel write channel-major for the next convolution or frame-major for the transformer — HF's permute after conv2, done by the store |
| `AttentionScores`, `AttentionApply` | the two halves of multi-head attention with `Softmax` between them. Query and key counts are separate uniforms, so cross-attention and a KV cache are the same kernels, and a `causal` flag masks the keys later than each query — query `i` standing at `keyCount - queryCount + i`, which is a cache's own shape and the encoder's square lower triangle at once. The head is folded into the dispatch row, since eacp dispatches 1D or 2D and no further (see the gaps section) |
| `Add` | the residual and positional sums |
| `Embed` | the decoder's input row: the token and positional gathers and their sum in one store. Ids travel as uint32 and are read back with `asUInt`, because a vocabulary is indexed and not measured |
| `Argmax` | greedy sampling's pick, with a caller-supplied mask taking Whisper's suppressed tokens out of the running before the comparison rather than after it. Lowest index on ties, which is what `torch.argmax` documents |
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

### `Decoder/` — `whisper-decoder`, `Tests/Decoder`

HuggingFace's `WhisperDecoder.forward` with a KV cache, out of the same kernels
and reached by the same loader. `DecoderShape` is the config plus the encoder's
row count — a parameter for the reason `EncoderShape::inputFrames` is one — and
`DecoderWeights` loads every tensor under its HF name. There is no
`proj_out.weight` in the file: the logits are `h · embed_tokensᵀ`, tied to the
input embedding, which is why `embed_tokens` is the one tensor that must arrive
F32 whatever the repo ships.

A run is a `beginSequence` and then `step`s. `beginSequence` projects the
cross-attention keys and values out of the encoder's rows once per layer — they
do not depend on the tokens — and resets the position; a `step` appends however
many tokens it is given, so the prompt is one call and each token after it is
another. Buffers are sized once for the longest step, which is what lets those
two be the same dispatches at different heights.

The KV cache is the reason the `BufferRange` gap above was filled: the key and
value projections write **into the cache at the step's own row** through a range
bind, and the attention that follows reads the whole cache. Without it a step
would need a scratch row and a copy per layer per token, or a row-offset uniform
inside `Linear` that its every other call site would pay for.

| test | tolerance | worst error measured |
| --- | --- | --- |
| small shape, a four-token prompt in one step | 5e-6 | 3.6e-7 |
| the same tokens one at a time through the cache | 5e-6 | 3.6e-7 |
| a prompt, single steps, then a second sequence on another encoder output | 5e-6 | 4.1e-7 |
| the same with fp16 projections | 5e-6 | 1.8e-7 |
| tiny.en over 4 tokens and 32 encoder rows, all 4 layers | 1e-4 | 5.4e-5 |

in the mixed measure `|a - e| / (1 + |e|)`, against a double-precision decoder
written in the test from HuggingFace's definition. The `Argmax` kernel's greedy
pick is asserted against the reference's in every one of them, which puts it at
the real 51864-wide row.

A single-token step at the shape it is really run at — 1500 encoder rows, four
layers, the full vocabulary — is about 9 ms of wall clock around `commit()`
(4.6 to 13.7 over repeated runs from a Debug host build, which is the only clock
available), after about 5 ms to project the cross-attention keys and values for
the whole sequence. Naive per-thread kernels with no tiling, as everywhere else.

Mutation-checked rather than trusted green: running cross-attention first,
projecting k and v from the unnormalised hidden state, binding the cache one row
out, and embedding at position zero each fail the suite. Dropping the causal
flag fails five of the six and *has* to leave the one-token-at-a-time test
standing — one query against n cached keys masks nothing, since the query stands
at the last of them.

What is deliberately not in here is `Argmax`. Which tokens are suppressed at
which step is generation config, so greedy sampling is the layer above.

### `Whisper/` — `whisper-runtime`, `Tests/Whisper`

`WSP::Whisper` owns the config, the filterbank, both weight sets, the
tokenizer, the mel front-end, the encoder, the decoder, `Argmax` and the two
suppression masks. `load(directory)` reads the four HF files and names the
missing one in a `ModelError`; `prepare()` is the GPU half. `transcribe` takes
16 kHz mono samples up to 30 s, zero-fills to the window as HF does, and
returns the sampled ids; `transcribeText` reads them back through the
tokenizer with the specials dropped. More than 30 s is an error for now —
chunking is a layer above this one.

The search is HF's greedy `generate`: the prompt is `<|startoftranscript|>`
then `<|notimestamps|>` in one step, `suppress_tokens` masks every sampled
step, `begin_suppress_tokens` (a blank and end-of-text) only the first, and
timestamps are not masked — HF adds its timestamp processor only when
timestamps are asked for, and the prompt token is enough, which a mutation
proves: drop it and the transcript fills with timestamps. whisper.cpp differs
three ways (`whisper.cpp:6194`): with `no_timestamps` it masks every timestamp
id at every step, it unconditionally masks `<|notimestamps|>` and the task and
language tokens, and its non-speech list — HF's `suppress_tokens` — defaults
to off, so a default whisper.cpp run suppresses less than a default HF one.
Its per-segment cap of 220 tokens is a chunking policy and is not adopted; the
cap here is HF's 448 positions.

Each decode step is its own committed command buffer, because the sampled id
has to reach the host before the next `Embed` can read it: eacp has no integer
input buffer and `write()` takes a `UInt` only into an `AtomicBuffer`, so no
kernel can hand a token to the next step on the device. That is the whole of
the per-step cost below.

On `jfk.wav`, tiny.en produces 24 tokens:

```
 And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.
```

with the leading space and no comma after "Americans", which is what the model
says and is pinned as a committed fixture (tier 3). From a Debug host build,
wall clock around `commit()`:

| stage | time |
| --- | --- |
| load and upload | 0.3 to 0.45 s |
| mel and encoder | 53 to 66 ms |
| decoding, 25 command buffers | 0.35 s, 14 ms per step |
| mel to transcript | 0.42 s |

`Apps/Console/Transcribe` runs the same on a WAV file and prints those. The
WAV reader is `Audio/WavFile`, PCM16 or float, first channel, refusing any
rate but 16 kHz; MakeASound links miniaudio privately and exposes no decoder.

Mutation-checked: stopping one token early and dropping either prompt token
fail the suite. Dropping the first-step mask is invisible on `jfk.wav` — and
on eleven other signals swept for it: silence, noise, clicks, DC, the sample
reversed, quieted and padded all come out word for word the same. A pure 440 Hz
tone is the one that shows it, answering end-of-text at once and producing an
empty transcript, so that is the test.

## Embedding the model

`WHISPER_EACP_EMBED_MODEL` (default **on**) runs the four fetched files through
ResEmbed into `whisper-embedded-model`, under category `WhisperModel` and keyed
by their HF basenames, which are the names `loadEmbedded()` asks for. Embedding
needs the files, so the fetch condition is `WHISPER_EACP_FETCH_MODEL OR
WHISPER_EACP_EMBED_MODEL` — a default configure downloads 151 MB. Off, the
target is still there as an empty `INTERFACE` library, so a consumer links it
unconditionally and `Whisper::hasEmbeddedModel()` is what answers, the same
shape as a test skipping on a missing file rather than on an `#ifdef`.

The API under it:

- `SafeTensors` now has three constructions — mapped, owned, and **borrowed**
  (`fromView`). The borrowed one is what makes an embedded model free: the
  151 MB the binary already carries is read where it lies, never copied to the
  heap.
- `Whisper` has a second `load`, over a `ModelFiles` of four byte spans, plus
  `loadEmbedded()` and `hasEmbeddedModel()`. Both loads run the same four
  parses in the same order, so a set of bytes missing what a directory would
  have been missing fails with the same message.
- `Audio/` decodes a WAV from a span — `readWavBytes(bytes, name)`, with
  `readWavFile` written on top of it — so an embedded recording and a file on
  disk report a refused sample rate identically.
- `whisper-runtime` links ResEmbed **PRIVATE**, through eacp's own find module.
  There is deliberately no `FindResEmbed.cmake` here: `CMAKE_MODULE_PATH` is
  inherited by every subdirectory and ours is appended first, so a module of
  that name would shadow eacp's.

The sample moved the other way. `Samples/jfk.wav` is committed — 352 kB, and
the only recording the suite needs — so `whisper_fetch_sample_file` and the
`whisper-sample-jfk` package are gone and `WHISPER_EACP_SAMPLE_DIR` is set in
the root `CMakeLists.txt`. It stays separate from the model directory for the
reason it always was: `WHISPER_MODEL_DIR` may point at a HuggingFace checkout,
and jfk.wav is not in one.

`Apps/Console/Transcribe` takes three forms — no arguments (embedded model,
embedded sample), one (embedded model, your WAV), two (a HF directory and your
WAV) — and names the model and the audio it chose before the transcript, so a
run says which form it took. `Tests/Embedded` asserts that
`hasEmbeddedModel()` equals the build flag (the one test here that never skips:
it is what catches ResEmbed's static-initializer registration being
dead-stripped, which would otherwise turn every other test in the file into a
silent skip), that each embedded view is byte-identical to the fetched file,
and that a runtime loaded out of the binary produces the pinned transcript.

## Validation, in the order a failure is diagnosed

1. **A scalar CPU reference inside the test itself**, for every kernel. No
   dependency, runs anywhere, and the only tier that catches a backend
   divergence — the same assertion runs against MSL here and HLSL on Windows.
   This is the bulk of the suite.
2. **whisper.cpp as an in-process oracle**, wired up behind
   `WHISPER_EACP_ENABLE_WHISPER_CPP` (default off) in `Tests/Oracle`. Stage-level,
   so a mismatch bisects to a layer instead of reporting a wrong transcript.
   What it found is below.
3. **Small committed fixtures** for what is cheap and stable.

A GPU test returns early when `Device::shared().isValid()` is false, so the
suite still passes on a machine with no GPU.

### The oracle

whisper.cpp `v1.9.3`, fetched with CPM and built CPU-only — `GGML_METAL`,
`GGML_OPENMP`, `GGML_ACCELERATE` and `GGML_BLAS` all off, `BUILD_SHARED_LIBS`
held off through CPM's CMP0077, and `CMAKE_BUILD_TYPE` saved and restored
around the fetch because whisper.cpp force-sets it in the cache. Four targets
and under three seconds from clean. Its `ggml-tiny.en.bin` (75 MB) is fetched
beside it, hash pinned, and `tokenizer.json` joined the HF fetch list so the
tokenizer can be compared at all. Every oracle test skips without the files.

It goes both ways: whisper.cpp's `_LIBCPP_REMOVE_TRANSITIVE_INCLUDES` breaks
ggml's own `gguf.cpp`, so the directory's compile definitions are cleared
around the fetch and restored after.

**The front-end, isolated through whisper.cpp's own encoder and decoder.**
whisper.cpp does not expose its mel or its encoder output, but it accepts a mel
and exposes logits, so the comparison is its logits from its own
`whisper_pcm_to_mel` against its logits from our `MelSpectrogram` handed in
through `whisper_set_mel`, with everything after the mel being whisper.cpp both
times. On `jfk.wav`, zero-filled to 30 s, the rows differ by 0.036 at most
(rms 0.013, logits ranging up to 20.8), and by 0.027 on a synthetic tone plus
noise; the argmax agrees on both. The assertion is not a number picked by hand:
the test moves every mel cell by one ulp and measures how far the reference's
own row moves (0.04), and asserts ours is within twice that. The response
saturates at once — 1e-7, 1e-6 and 1e-5 perturbations all move the row by
about 0.04 — so that is the floor of what fp16 GGML weights let the comparison
resolve, and our front-end sits at it.

Format conversion ruled out first, as CLAUDE.md asks: the filterbank inside
`ggml-tiny.en.bin` is bit-identical to `preprocessor_config.json`'s, all 16080
entries. The residual is STFT arithmetic — their float radix-2 FFT against our
per-bin DFT on the GPU. Two deliberate differences are documented in the test:
whisper.cpp pads the tail with 30 s of zeros where HF reflects, which is why
both signals are zero-filled to exactly 480000 samples so the two agree; and it
frames 6000 windows of which the encoder reads 3000.

**The two transformer stacks, at the logits.** Our mel through our encoder and
decoder against the same mel through whisper.cpp's, for the prompt and four
further steps fed the reference's own pick each time: the argmax agrees at
every step, and the rows differ by 0.47 at most (rms 0.27) against the 0.04 the
reference's one-ulp resolution gives — 11.6 times it, asserted at 25. The
residual is the GGML fp16 conversion rather than ours: `Tests/Decoder` has the
same decoder within 5.4e-5 of a double-precision reference on the same weights.

**The transcript.** `transcribe` against `whisper_full` on `jfk.wav`, greedy,
no timestamps, one segment, with whisper.cpp's non-speech suppression turned on
so both sides run the same policy: 24 tokens to 24, token for token, and the
joined text byte-identical.

**The tokenizer.** Decode agrees on every id tried and every special token,
and the vocabularies are the same size. Encode agrees on 7 of 8 strings; the
one that differs, `"Le café était naïf."`, is ours that is right — byte-level
BPE recomputed independently from `tokenizer.json`'s merge ranks gives our
sequence, and whisper.cpp's `tokenize()` is not BPE but a greedy longest-match
over a `std::regex` whose `[[:alpha:]]` is ASCII where GPT-2's pattern is
`\p{L}`. Both segmentations decode back to the input, which is what the test
asserts unconditionally; exact agreement is measured and printed.

## Gaps found in eacp, Miro and ResEmbed

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
| ~~`Span::size()` is `int`; `MemoryMappedFile::bytes()` is not~~ **written down** | `MemoryMappedFile.h:51` said the size is the full `size_t` "as is `bytes().getSize()`", but `Span::size()` asserts `fitsInt` (`ea_data_structures/Structures/Span.h:168`), so `mapped.bytes().size()` aborts in Debug and truncates in Release on exactly the file the class exists for. `Model/` uses `getSize()`, and eacp `0db26a3` puts the sentence beside `bytes()` |
| ~~`MemoryMappedFile` is not move-assignable~~ **filled** | `Pimpl` (`Core/Utils/Pimpl.h:17`) defaulted its move constructor and declared no `operator=`, so `std::optional<MemoryMappedFile>` took `emplace()` but rejected assignment, which is surprising for a type its header says to "move". eacp `0db26a3` defaults `Pimpl& operator=(Pimpl&&)`, with a test in eacp's `Tests/Core` assigning over a live optional. What that leaves in view is that a moved-from `Pimpl` holds a null `shared_ptr`, so `->` on one is undefined — pre-existing, now reachable one more way, and the maintainer's call |
| ~~`readHalf` reads a whole word~~ **written down** | `ShaderValue.h:2502` fetches `[index / 2]`, so a buffer holding an odd number of halves is read one word past its last whole word on the final element. The loader rounds the allocation up, and eacp `0db26a3` says so beside `readHalf` and in the README's fp16 section |
| no `graph()` on a shipped program | `ShaderProgram` exposes `source()` for the active backend but not its graph, so pinning the *other* backend's text for a real kernel means re-authoring its body on a bare `ShaderBuilder` — which is how eacp's own tests do it. An accessor would let a shipped kernel's HLSL be asserted on a Mac |
| loop-invariants are recomputed | the emitter gives up open names at a loop header by design (`StageEmitter`), so softmax's `1.0 / total` is divided once per element rather than hoisted. Harmless there; a real cost for a loop whose invariant is expensive and provably unwritten |
| `erf(0)` is 1e-9 | eacp's helper ends in `x < 0 ? -e : e` where the deleted local one ended in `sign(x) * (...)`, so it is not odd-symmetric at the origin. GELU multiplies it by `x == 0`, so nothing here sees it; a caller wanting `erf` itself would |
| ~~`ComputePass` binds whole buffers only~~ **filled** | `Frame/ComputePass.h:51` took a `Buffer` and never the `BufferRange` that exists (`Buffer/Buffer.h:138`) and that `RenderPass::setVertexBuffer` already took. The encoder got away with it because `embed_positions` is read as a prefix from offset 0; a KV cache has no way to bind at an offset short of a second buffer or an index uniform. eacp `develop` from `0db26a3` adds `setInputBuffer`/`setOutputBuffer` over a range and the three `Uniform<...Buffer>` assignments, with `Tests/GPU/ComputeBufferRangeTests.cpp` pinning both directions, the atomic forwarding, the four-byte stride the alignment rests on, and that an invalid or out-of-range bind binds nothing. The decoder's k and v projections write straight into the cache row through it |
| no stated contract for aliasing an input and an output | nothing says whether a hoisted read survives a store through a differently declared pointer to the same allocation, so every 1:1 elementwise stage (`Gelu`, `Add`) writes to a separate buffer: 12 extra allocations, 108 MB of them the score and probability pair at the full window |
| GPU timing is `Frame`-only | `Timing/FrameTimer.h:15` is driven by `Frame`, and `CommandBuffer` has no timestamp hook, so an off-screen compute pass is timed by wall clock around `commit()` |
| no device-side zero fill | `Device::makeBuffer(int)` (`Device/Device.h:57`) allocates uninitialised, so k_proj's zero bias is a host vector uploaded once. Minor |

### What assembling the decoder surfaced

The third round, against eacp `develop` at `0db26a3`, which this project
fetches. The `BufferRange` bind above is the one gap the decoder could not do
without; the rest is what filling it and writing the two new kernels turned up.
None is worked around silently.

| finding | where, and what it means |
| --- | --- |
| no integer input buffer | `InputBuffer::operator[]` yields only `Float` (`Codegen/ShaderValue.h:582`), and a read binding is emitted as `device const float*` / `StructuredBuffer<float>` (`Codegen/ShaderEmitter.cpp:1432`, `:1447`). The only uint-element buffer is `AtomicBuffer` (`ShaderValue.h:729`), which binds as an output (`Codegen/ComputeProgram.h:79`) — a read-write binding for what is an input. `Embed` uploads uint32 ids and reads them back through the `asUInt` bitcast (`ShaderValue.h:2415`), which is exact by construction; every Whisper id below 2^23 is a subnormal float pattern, so a test gathers ids 0, 1, 220, 50256, 50257 and 51863 and gets each back exact on Metal |
| `write()` takes a `UInt` only into an `AtomicBuffer` | `Codegen/ComputeProgram.h:337`; the `OutputBuffer` overloads are `Float`/`Float2`/`Float3`/`Float4`. So `Argmax` writes its index through `atomic_store_explicit` on a `device atomic_uint*` — an atomic for a store no two threads contend. The same gap as the row above from the write side: eacp has a float storage buffer and an atomic uint buffer, and no plain integer one |
| `select` is float-only | constrained through `SameShaderShape` → `detail::baseOf`, declared for the four float shapes only (`ShaderValue.h:869`, `select` at `:1898`), while `min`/`max` *are* defined on `UInt` (`:2649`). `Argmax` keeps its running best in `Var<UInt>` under `ifThen`, which is the right tool for it anyway; the causal mask selects on `Float` and is unaffected |
| a hoisted name does not cross a branch | the emitter names a repeated subexpression as a `tN` local but does not carry it into an `ifThen` body, so `Argmax`'s `logits[base + index]` is loaded once in the condition and once in the body. Two loads per element in a one-thread-per-row scan; left as written to match `Softmax` next door |
| `RenderPass::setVertexBuffer(const BufferRange&)` guards differ by backend | `Frame/RenderPass-Windows.cpp:168` rejects an offset at or past the end; `RenderPass-Apple.mm:232` has no offset guard at all, so a negative or past-end offset reaches Metal. The new compute binds guard on both, so they agree with each other and not with the render path |
| no ranged render-side storage bind | `RenderPass::setVertexStorageBuffer` / `setFragmentStorageBuffer` (`Frame/RenderPass.h:192`) take only a `Buffer`, so a `Uniform<InputBuffer>` on a `ShaderProgram` binds whole; rather than drop an offset silently, `ShaderBufferBindVisitor::onInputBuffer` now asserts `offset == 0` in Debug. The obvious next range gap, not needed here |
| `dispatchIndirect` does not bound its offset | `ComputePass-Apple.mm:156` and `ComputePass-Windows.cpp:215` both reject only `offsetInBytes < 0`, never one at or past `arguments.size()`, unlike the binds beside them |

Checked and not a gap: an unassigned `Uniform<UInt>` is zero — `Uniform<T>` holds
`Cpu value {}` (`Codegen/ShaderProgram.h:245`) and `CpuValueOf<UInt>` is
`std::uint32_t` — so the encoder's `AttentionScores` would have masked nothing
without the explicit `causal = 0u` it sets anyway.

The alignment the ranged bind documents is four bytes on both backends, from
what the emitter declares: `device float*` and `device atomic_uint*` on Metal,
`StructuredBuffer<float>` / `RWStructuredBuffer<float>` / `RWStructuredBuffer<uint>`
on HLSL, never a `ByteAddressBuffer` with its sixteen-byte rule — and an eacp
test asserts the emitted declarations that number rests on. `range.bytes` is
enforced by neither backend, which the header says outright.

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

### ResEmbed

Embedding the model is the first thing here to hand ResEmbed (eacp fetches it;
`0ea662c`) a resource that is not a font or an icon, and 151 MB is where its
choices start to show. All three belong upstream; none is worked around here.

| finding | where, and what it means |
| --- | --- |
| bytes are emitted as a decimal brace initializer | `Generator/ResourceGenerator.cpp:83`, `generateDataFile` — `out << static_cast<unsigned int>(data[i])` with a comma between, 16 per line. `model.safetensors` at 151,041,024 bytes becomes a **656,571,546-byte `.c`**, which clang compiles in **41.6 s** at a **16.2 GiB peak RSS**, for a 151 MB `.o`. That is 4.3 bytes of source per byte of data and roughly 115 bytes of compiler memory per byte of data, and it is the whole cost of the option. A string literal with hex escapes would cost about the same source and far less memory; `.incbin` on the assemblers that have it, or C23 `#embed`, would cost neither. The README already argues the C front end over C++ for exactly this reason, so the direction is agreed and only the encoding is the gap |
| `ResEmbed::get` default-inserts on a miss | `Lib/ResEmbed/ResEmbed.cpp:72` is `Detail::getMap()[category][name]` — `std::map::operator[]`, so a lookup that finds nothing *writes* an empty `DataView` and, if the category is new, a whole `ResourceMap` under it. `Whisper::hasEmbeddedModel()` on a binary that embeds nothing therefore grows the registry by a category and four entries, under the mutex, every time it is asked. It answers correctly — an inserted `DataView` is empty and `operator bool` is `!empty()` — so nothing here is wrong, but a query is not a mutation. A `find`-based lookup, or a `contains(name, category)` beside `get`, is the API this wanted. `getCategory` (`:78`) is the same call |
| `DataView` has no `std::span` accessor | `Lib/ResEmbed/ResEmbed.h:36` stores a `View`, which is `std::span<const unsigned char>` (`Common.h:8`), and exposes it only as `data()` (`:22`) plus `size()`/`getSize()` (`:29`, `:30`). So converting one to `Span<const std::uint8_t>` reassembles the pointer and the length the object is already holding, and leans on `std::uint8_t` being `unsigned char` — true everywhere this builds, and not something the type says. A `view()` returning the member would make the conversion exact rather than merely correct |

### Checked and discarded

`eacp::MemoryMappedFile` **does** exist — commit `3081829`, compiled into
`eacp-core`, `bytes()` returning a span — and `Model/` now maps the weights
file through it. What that turned up is the `Span::size()` pairing in the table
above.

## After the transcript

The pipeline runs end to end: 30 seconds of audio in, a string out, checked
against a double-precision reference at every layer and against whisper.cpp at
the mel, the logits and the transcript. What is above it now:

- **Chunking.** More than 30 s is an error today. Whisper's own answer is to
  decode a window, take the last timestamp as the seek point, and continue —
  which means decoding *with* timestamps, so the timestamp tokens the search
  leaves unmasked start to matter, and the `<|startofprev|>` prompt carries the
  previous window's text across.
- **Streaming through MakeASound.** `Audio/` captures; nothing feeds the
  capture into a window yet. The encoder is 60 ms and a step 14 ms, so a
  window re-decoded as it fills is affordable, but the step's host round trip
  is the thing to remove first: a device-side token buffer between steps is
  exactly the integer buffer eacp does not have (the gaps table).
- **Multilingual prompts.** The language and task tokens between
  `<|startoftranscript|>` and `<|notimestamps|>`, and language detection from
  the first step's logits over the language ids. `SpecialTokens` already names
  them; `Whisper` does not emit them.
- **Resampling** in the WAV reader and the capture path, since 16 kHz mono is
  the model's contract and not the microphone's.
- **Kernels.** Every kernel is still the naive per-thread version the scalar
  references were written against. Tiling the two matmul shapes is where the
  time is.

Two things ours surfaced on the way, recorded here rather than in the
dependency tables: `SafeTensors::makeBuffer` and `PreprocessorConfig`'s
filterbank upload take no `Device` and use the shared one, so
`Whisper::prepare(device)` compiles on the given device and uploads on the
shared one; and `EA::Span`'s deleted rvalue-container constructor
(`Structures/Span.h:158`) makes `f(g())` a compile error whenever `g` returns
a `Vector`, which is deliberate and costs a named local at every such call —
`readWavBytes(readWholeFile(path), name)` inside `Audio/WavFile.cpp` and
`transcribe(readWavFile(...))` in `Tests/Embedded` are two more of them, and
each reads as a temporary that had to be given a name for no reason a call
site can see.
