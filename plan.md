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

in the mixed measure `|a - e| / (1 + |e|)`. The full 30 s window gave
1500 x 384 finite values in about 60 ms of GPU time on this machine from the
naive per-thread kernels with no tiling; the tiled kernels of the performance
round below do it in about 7 ms, mel excluded.

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
layers, the full vocabulary — was about 9 ms of wall clock around `commit()`
from the naive per-thread kernels, after about 5 ms to project the
cross-attention keys and values for the whole sequence. It is 0.9 ms after
the performance round below.

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

Each decode step was its own committed command buffer, because the sampled
id had to reach the host before the next `Embed` could read it: eacp had no
integer input buffer then. It has one now, and the sequence lives in one uint
buffer on the device — the prompt uploaded into its first slots, every slot
after them written by one step's `Argmax` and read by the next step's `Embed`
through ranged binds — so steps go four to a command buffer and the host
reads a slot back only to learn whether the run is over. The numbers below
are from before that.

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

## Bundling the model

The model reached the binaries in two steps, and the second undid the first.

**First, embedded.** `WHISPER_EACP_EMBED_MODEL` ran the four fetched files
through ResEmbed into a `whisper-embedded-model` target, under category
`WhisperModel` and keyed by their HF basenames, with `loadEmbedded()` and
`hasEmbeddedModel()` reading the registry. It worked, and it is what surfaced
the three ResEmbed findings below — the first of which is the whole cost of the
approach: 151 MB as a decimal brace initializer is a 657 MB `.c`, a 42 s compile
and a 16 GiB peak RSS, for every binary that wanted the model.

**Then, copied.** The embedding is gone. `WHISPER_EACP_FETCH_MODEL` (default
**on** now, and the only model switch) fetches the four files, and
`whisper_bundle_model(<target>)` copies them beside a binary after every link,
under a `WhisperModel` directory: into `Contents/Resources` of a `MACOSX_BUNDLE`
target, and next to the executable otherwise — a Windows build, or a macOS
executable that is not a bundle, which the console apps and every test are. The
generic half is `whisper_copy_resources(<target> FILES ... [DESTINATION <dir>])`
in `CMake/WhisperResources.cmake`, a `POST_BUILD` `copy_if_different`; the
model-specific half is defined in `Model/CMakeLists.txt` beside the fetch,
exists whether or not the fetch is on, and does nothing when it is off, so a
consumer calls it unconditionally. Nothing is compiled and nothing is linked.

The runtime resolves the same rule from inside the process.
`WSP::resourcesDirectory()` (`Whisper/ResourcesDirectory.h`, one source per
platform) is CoreFoundation's `CFBundleCopyResourcesDirectoryURL` on Apple —
which answers the executable's own directory for a bundle-less binary, the
detail that lets a console tool and an `.app` share one rule — and the parent of
`GetModuleFileNameW` on Windows. `Whisper::bundledModelDirectory()` appends
`WhisperModel`, `hasBundledModel()` checks the four files are in it, and
`loadBundled()` is `load(path)` on it, so the weights are memory-mapped exactly
as for a directory named on the command line. The directory name is spelled in
`Model/CMakeLists.txt` and `Whisper::bundledModelDirectoryName`, and nowhere
else.

What the first step built and the second kept:

- `SafeTensors` has three constructions — mapped, owned, and **borrowed**
  (`fromView`). The borrowed one was what made an embedded model free, and it
  stays as the path for weights a caller already holds in memory.
- `Whisper::load(const ModelFiles&)`, over four byte spans, runs the same four
  parses as `load(path)` in the same order, so a set of bytes missing what a
  directory would have been missing fails with the same message.
  `Tests/Whisper` loads through it.
- `Audio/` decodes a WAV from a span — `readWavBytes(bytes, name)`, with
  `readWavFile` written on top of it. `Transcribe`'s sample is still embedded
  through ResEmbed: 352 kB is what ResEmbed is for, and it is the one
  `find_package(ResEmbed)` left in the tree. There is deliberately no
  `FindResEmbed.cmake` here: `CMAKE_MODULE_PATH` is inherited by every
  subdirectory and ours is appended first, so a module of that name would
  shadow eacp's.

The sample moved the other way. `Samples/jfk.wav` is committed — 352 kB, and
the only recording the suite needs — so `whisper_fetch_sample_file` and the
`whisper-sample-jfk` package are gone and `WHISPER_EACP_SAMPLE_DIR` is set in
the root `CMakeLists.txt`. It stays separate from the model directory for the
reason it always was: `WHISPER_MODEL_DIR` may point at a HuggingFace checkout,
and jfk.wav is not in one.

`Apps/Console/Transcribe` takes three forms — no arguments (bundled model,
embedded sample), one (bundled model, your WAV), two (a HF directory and your
WAV) — and names the model directory and the audio it chose before the
transcript, so a run says which form it took. `Tests/Bundled` (was
`Tests/Embedded`) asserts that the resources directory the runtime resolves is
real (never skipping, on both platforms), that the copy is there when the build
made one, that each copied file is byte-identical to the fetched one, and that
a runtime loaded out of the copy produces the pinned transcript. The one
assertion the embedded version had that this one does not is "no model when
the option is off": a `POST_BUILD` copy leaves its output behind when a build
directory is reconfigured from on to off, and a test that read that as a
failure would be reporting a stale build tree, not a bug.

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

### What bundling the model surfaced

One, in eacp, and it is the reason `Whisper/` has platform sources at all:

| finding | where, and what it means |
| --- | --- |
| `Files::getBundleResourcePath` is Apple-only, and there is no accessor for the directory itself | `Core/Utils/Files.mm` resolves a named resource through `CFBundleCopyResourceURL`; `Files-Windows.cpp` and `Files-Linux.cpp` return `{}` for every name. "Beside the executable" is the Windows analogue of a bundle resource — it is what `whisper_copy_resources` does there — so an app that ships a file with itself has no eacp call that finds it on both platforms. Nor is there a `resourcesDirectory()` returning the directory rather than a file in it, which is what a caller wants when the resource is a directory of four files and the error message should name where it looked. `Whisper/ResourcesDirectory-Apple.cpp` and `-Windows.cpp` are the two answers, written here; the Apple one is `CFBundleCopyResourcesDirectoryURL`, which already does the right thing for a bundle-less executable |

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

Embedding the model was the first thing here to hand ResEmbed (eacp fetches it;
`0ea662c`) a resource that is not a font or an icon, and 151 MB is where its
choices start to show. All three belong upstream; none is worked around here.
The first row is why the model is no longer embedded at all — *Bundling the
model* above — and the sample, at 352 kB, still is.

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
  capture into a window yet. The encoder is 10 ms and a step under 1 ms, so a
  window re-decoded as it fills is affordable.
- **Multilingual prompts.** The language and task tokens between
  `<|startoftranscript|>` and `<|notimestamps|>`, and language detection from
  the first step's logits over the language ids. `SpecialTokens` already names
  them; `Whisper` does not emit them.
- **Resampling** in the WAV reader and the capture path, since 16 kHz mono is
  the model's contract and not the microphone's.
- **Kernels.** What the performance round below left: the decode step is a
  chain of some sixty small kernels whose latencies add up, and the fix for
  that is concurrency in eacp's compute pass rather than another kernel here.

### Where it stood against whisper.cpp

`Benchmark/` is the baseline the kernel work is measured against: our runtime
and whisper.cpp v1.9.3 in one Release process, on the same 480000 samples,
the same greedy policy, one warm-up and the median of ten timed runs. Behind
`WHISPER_EACP_ENABLE_BENCHMARK`, which is also what decides which whisper.cpp
a tree gets — its targets are global, so there is one — and the benchmark's
wins: whisper.cpp at its own defaults, Metal, Accelerate and BLAS on, where the
oracle alone builds it with every backend off. It runs once on Metal and once
without it, at its default four threads. The three transcripts agree token for
token.

Apple M4 Max, `tiny.en`, `jfk.wav` zero-filled to the window:

| | WhisperEACP | whisper.cpp Metal | whisper.cpp CPU |
| --- | --- | --- | --- |
| transcribe, median | 0.393 s | 0.040 s | 0.124 s |
| x real time, over the 30 s window | 76 | 756 | 242 |
| encode, median | 0.046 s (mel + encoder) | 0.008 s (encoder only) | 0.087 s |
| decode, median, 25 steps | 0.345 s | 0.016 s | 0.021 s |
| decode per step | 13.8 ms | 0.6 ms | 0.8 ms |

Two readings, at the time. The encoder was six times whisper.cpp's on the
same GPU, and that gap was the naive kernels: the two matmul shapes were the
whole of it. The decoder was twenty times, and a Release build moved the step
from the 14 ms the Debug tests recorded to 13.8 ms — so the step was not host
code but the GPU side of the loop. What each turned out to be is the section
after next.

`Benchmark/backends.py` runs mlx-whisper, faster-whisper and openai-whisper on
the same protocol for the runtimes that cannot be in the process. Written
against their documented APIs and not yet exercised: none of the three is
installed on this machine.

### The performance round

The same benchmark, the same machine, after the kernels were rewritten for
speed. Every kernel keeps agreeing with the scalar references, the
double-precision encoder and decoder references and whisper.cpp's logits and
transcript — 248 tests — and every change was measured on the way in with
eacp's labelled passes, which time the GPU per pass.

| | WhisperEACP | whisper.cpp Metal | whisper.cpp CPU |
| --- | --- | --- | --- |
| transcribe, median | 0.032 s | 0.037 s | 0.103 s |
| x real time, over the 30 s window | 932 | 822 | 292 |
| encode, median | 0.010 s (mel + encoder) | 0.008 s (encoder only) | 0.073 s |
| decode, median, 25 steps | 0.022 s | 0.015 s | 0.015 s |
| decode per step | 0.9 ms | 0.6 ms | 0.6 ms |

The first profile of a decode step said where the 9.4 ms went, and it was
not where the previous section guessed: 3.9 ms in `Argmax`, one thread
walking 51864 logits; 1.5 ms in four softmaxes over 1500 keys, one thread per
row; 1.6 ms in four attention applies, one thread per column walking 1500
keys; 0.9 ms in thirteen layernorms over a single row; 1.4 ms in thirty-three
matrix-vector products, one thread per output walking 384 or 1536 inputs. The
vocabulary projection, 80 MB of F32 read once, was 0.2 ms and already at the
bandwidth. Every one of those is a loop whose depth is the cost, and the
whole of the encoder's 46 ms was the same shape: a thread per output of a
1500-row product.

What changed, in the order it was measured:

| change | what it is | effect |
| --- | --- | --- |
| `Reduce.h`: group-per-row `Softmax`, `LayerNorm`, two-stage `Argmax` | 64 lanes walk a row's strided shares and fold in shared memory; the argmax spreads a row over ~100 groups and folds (value, index) pairs, lowest index on ties, into a `UIntOutputBuffer` | step 9.9 to 1.2 ms with the two rows below |
| `SplitLinear` | a step's matrix-vector products with the inner sum split eight ways across a group, `read4` along the weight row | in the row above |
| split-key `AttentionApply` | the same split over the keys | in the row above |
| `TiledMatMul` | 32 x 32 tiles of C per group, a 4 x 4 block per thread in registers, 16-deep slabs of A and B in shared memory; strides and a batch fold so a head's slice of an activation is an operand; `forLinear`, `forAttentionScores` (scaled, causally masked on the store) and `forAttentionApply` name the three products; fp16 weights through the same template | encoder 46 to 13 ms; 3.8 TFLOPS on a 384-wide projection, 5.2 on fc1 |
| gelu and residual folded into the stores of `SplitLinear` and `TiledMatMul` | the stages either side of every projection, gone as dispatches | 16 fewer dispatches a step, 5 a layer |
| `Unfold` + `TiledMatMul` for both convolutions | PyTorch's unfold, the input addressed by two strides so the band-major mel and the frame-major activation are the same kernel; the weight in its file layout is the B operand | conv2 2.3 to 0.3 ms |
| `StftFramesKernel` + `TiledMatMul` against `Mel/Basis.h` | the windowed, reflect-padded frames as rows, the DFT as a [402, 400] matrix computed once, the power squared out of the pairs | STFT 3.4 to 0.5 ms; encoder 13 to 10 ms |
| `SingleQueryAttention` | a decode step's attention as a partial and a combine: eight groups per head over chunks of the keys, each with the query slice in registers, a chunk maximum, an exponential sum and an unnormalised row; the combine rescales the chunks by exp of their maxima less the largest, which is exact | step 1.0 to 0.9 ms; a single group per head was 1.6 ms, latency again |
| the sequence on the device, four steps a commit | `Embed` reads a `UIntInputBuffer`, `Argmax` writes a `UIntOutputBuffer`, both ranges of one buffer of slots | commit latency off the critical path; up to three steps computed past the end |
| one compute pass per step and per encode | every recording call takes the `ComputePass` the caller opened | no measurable change — the cost was never the pass boundary |

Three measurements that decided things. First, the per-dispatch GPU cost:
a trivial 64-element kernel dispatched a thousand times in one pass is
1.6 us a dispatch, and about 1 us a pass, so a step's sixty-odd dispatches
cost 0.1 ms of overhead and the rest of its 0.8 ms of GPU time is the kernels
themselves — small kernels whose few microseconds each are their latency,
serialised. Second, the step's host side: encoding the dispatches is 0.03 ms,
the commit and the wait around 0.2 ms over the GPU's own time and sometimes
0.5 ms more when the GPU had idled, which is what batching the steps was for.
Third, an eacp semantic that cost an afternoon: a buffer read handle names
the element, not the value, and re-materialises after a store to its slot —
the in-place softmax computed `exp(exp(x - max) - max)` for its sum until the
exponential was held in a `var`. It is documented in eacp's README under "In
place" and is what makes in-place scaling read what was stored; it is stated
in `Softmax.h` so the next kernel does not rediscover it.

Not done, and why: a concatenated q/k/v projection would drop eight dispatches
a step but needs the loader to build a joined tensor; folding the layernorm
into the projections that consume it would drop thirteen and is worth about
50 us; and F16 for the token embedding would halve the 80 MB the vocabulary
projection reads every step, which is a numerical decision about a model
that ships F32.

### What the performance round surfaced in eacp

Against eacp `develop` at `209e735`, which this project fetches. The step's
remaining 0.3 ms over whisper.cpp is the first row; none of these is worked
around here.

| finding | where, and what it means |
| --- | --- |
| a compute pass is serial | `ComputePass` dispatches run one after another: Metal's serial encoder, and a UAV barrier after every dispatch on D3D12 (`ComputePass-Windows.cpp:162`). ggml-metal encodes with `MTLDispatchTypeConcurrent` and a memory barrier only where a node depends on the one before, so its independent nodes — a layer's q, k and v projections, the two attentions' out projections — overlap and their launch latencies hide behind each other. A concurrent pass with an explicit `barrier()` the recording code calls between dependent stages is the missing piece, and it is what the decoder's chain of sixty small kernels needs |
| `Buffer::read` waits for every submission | `Buffer-Apple.mm:77`: "waiting for the newest submission waits for every" one before it, and there is no `CommandBuffer::wait()`. So a step's token cannot be read while the next step is in flight — the read would wait for both — and the CPU cannot encode step k + 1 while the GPU runs step k. A per-command-buffer wait, and a read that trusts the caller to have waited, would pipeline the loop; batching the steps is the workaround that is not one, since it is what keeps the GPU fed either way |
| dispatches within a pass are ordered, and the README does not say so | the README says "let a pass end before beginning the one that reads what it wrote", and both backends in fact order the dispatches of one pass (the serial encoder; the barrier after each). Everything here now records a step or an encode into one pass on that behaviour, which should be a promise |
| a read handle re-materialises after a store to its slot | documented under "In place", and by design; but an expression built on the read and used after the store is recomputed from the stored value, silently. Worth a sentence beside the rule: hold what a store must not change in a `var` |
| sixteen labelled passes per command buffer | `GpuTimestamps::maxTimedPasses = 16`: a step is one pass now, and was ninety, so per-kernel profiling here labels one block of sixteen per run and takes six runs. A per-dispatch timestamp, or a larger cap, is what a profile of a real net wants |
| the group shape is fixed | 64 in 1D, 8 x 8 in 2D, 4 x 4 x 4 in 3D. `TiledMatMul` is written to an 8 x 8 group with a 4 x 4 block per thread and gets 4 to 5 TFLOPS; a 16 x 16 group with the same block would halve the shared-memory traffic per FLOP, and a 1D group of 256 would give the reductions eight elements a lane over a 1500-key row instead of twenty-four. A per-program group size is the knob |
| no SIMD-group reductions | `simd_sum`, `simd_max` and their kin exist in MSL and as wave intrinsics in HLSL SM6, and are what ggml's matrix-vector kernels reduce with; the EDSL has none, and FXC at `cs_5_0` has none to emit, so every reduction here is shared memory and six barriers |

Two things ours surfaced on the way, recorded here rather than in the
dependency tables: `SafeTensors::makeBuffer` and `PreprocessorConfig`'s
filterbank upload take no `Device` and use the shared one, so
`Whisper::prepare(device)` compiles on the given device and uploads on the
shared one; and `EA::Span`'s deleted rvalue-container constructor
(`Structures/Span.h:158`) makes `f(g())` a compile error whenever `g` returns
a `Vector`, which is deliberate and costs a named local at every such call —
`readWavBytes(readWholeFile(path), name)` inside `Audio/WavFile.cpp` and
`transcribe(readWavFile(...))` in `Tests/Bundled` are two more of them, and
each reads as a temporary that had to be given a name for no reason a call
site can see.

## Live transcription

Three pieces, in the order the audio moves through them.

`Audio/Capture` is the microphone: a `MakeASound::DeviceManager` opened at
Whisper's own 16 kHz, the chosen slice of the chosen device's channels averaged
to mono in the device callback, and the result pushed into an SPSC queue the
main thread empties with `drain()`. Nothing on the audio thread allocates,
locks or calls back into the manager — the mono block is one scratch sized once
at 8192 frames and a larger block is dropped and counted, and the notifications
are drained on the caller's thread rather than taken from the callback the OS
raises them on, which can already hold the device. The peak and the RMS of the
last block go out through two relaxed atomics, which is the whole of what a
meter needs.

`Whisper/LiveTranscriber` is the policy: samples in, a growing transcript out.
It quantises everything to 100 ms blocks and measures every clock in audio time
rather than wall time, so a run over a given recording is deterministic. A block
whose RMS is above **-40 dBFS** is speech; the open segment keeps 0.3 s of
lead-in silence and drops what came before the first speech block; it is re-run
through `Whisper::transcribe` every **0.5 s** of new audio; and it closes — into
`committed()`, the open text moving out of `pending()` — on **1.0 s** of
continuous silence after speech, or at **25 s**, which is the 30 s window less
room to grow. A segment holding less than 0.3 s of speech is never sent to the
model at all. At most one model run happens per `update()`, which is what makes
it safe to drive from a timer.

`Apps/Demo/LiveTranscribe` is the window over the two: an input device and a
channel pair from MakeASound's `UIDeviceManager` dropdowns, a dB-scaled level
meter, and the transcript with the closed segments in the theme's text colour
and the open one in the accent, since every word of it may still change. The
whole loop is one 30 Hz `Threads::Timer` on the message thread — eacp's GPU
layer is main-thread only and `Whisper::transcribe` blocks on its own commits,
so there is nowhere else for it to be.

### What the live loop surfaced in the dependencies

| finding | where, and what it means |
| --- | --- |
| no wrapped text, and no text area | `UI::Graphics` draws a run on one line (`drawText(text, baselineLeft)`) and places one in a box (`drawText(text, area, justification)`); neither wraps, and `TextEditor` is single line "on purpose", wrapping being "a layout problem the tier has not needed yet". So `TranscriptView` measures word by word and lays the lines out itself, and because a component cannot resize itself, the height that wrap came to has to be pushed back into the `ScrollPanel` by the panel that owns both. A `drawFittedText`, or a read-only multi-line label, is the missing piece |
| the GPU is main-thread only, and a run blocks | `GPU/README.md:6`, and `Whisper::transcribe` waits on every `commit()`. A live run therefore stalls the window for the length of the run — tens of milliseconds in Release, a good deal more in Debug — and the meter and the transcript both freeze for it. The fix direction is a `CommandBuffer::commitAsync` inside `Whisper` (a completion handler rather than a wait), or a run on a device queue of its own that the message thread polls |
| `AudioCallbackInfo::maxBlockSize` is not the block delivered | `MiniAudioDeviceManager::openStream` overwrites `config.maxBlockSize` with the device's *internal* period at its *native* rate, so a 48 kHz device opened at 16 kHz reports about three times the frames the callback actually receives. Anything sizing a scratch from it over-allocates by the resample ratio; `Capture` sizes a fixed 8192-frame scratch and reports `info.numSamples` as the block size instead |
| no resampler choice | `StreamConfig` has no resampler field, so a device that cannot open at 16 kHz goes through miniaudio's default converter with no way to ask for a better one. On this machine no input lists 16000 Hz, so every capture here is resampled — and still arrives at exactly 16 kHz mono, 6656 samples in 400 ms, block 512 |
| nothing reports what a stream resampled *from* | miniaudio knows (`device.capture.internalSampleRate`) and MakeASound does not surface it, so "native or resampled" can only be inferred from the enumerated `sampleRates` snapshot, which another app can invalidate between the enumeration and the open |
| `SPSCQueue` has no bulk operations | no `pushRange`/`popRange`, no `size()`, no `clear()`. A 512-frame block is 512 acquire/release pairs each way, and flushing stale audio before a re-open is a pop-until-empty loop |
| `setConfig` always starts the stream | `DeviceManager::setConfig` unconditionally starts, so "change the config and stay stopped" needs the caller to guard on `isRunning()` before every re-open, which `Capture::applySelection` does |
| `getDevices()` re-enumerates every call | `MiniAudio::DeviceManager::getDevices` refreshes its cache on each ask rather than on a device notification, and with ten inputs on this machine that is about **90 ms on the calling thread**. A panel polling it every two seconds, which is what MakeASound's own probe app does, therefore hitches; the answer is a cached snapshot invalidated by the notification that already exists |
| opening an input blocks until the permission prompt is answered | `Capture::start` → `ma_device_init` → CoreAudio's `AudioDeviceCreateIOProcID` sits in `mach_msg` while TCC decides, so an app whose microphone grant has not been given yet freezes on the thread that asked — the message thread, here. MakeASound has neither an asynchronous open nor a way to ask about the permission first |
| `UI::Button` has no enabled state | so "Start is disabled until the model is ready" is spelled as an accent colour and a click the handler swallows. `Checkbox` and `Slider` are the same |
| `EA::Vector` has no bulk append from a `Span` | `LiveTranscriber::push` copies element by element. Not worth changing on its own, but it is the second half of the `SPSCQueue` row: nothing in either container library moves a run of samples in one call |

### After the live loop

Four things the live loop wants that are this project's own next work rather
than anything a dependency owes it, all of them in `Whisper`'s API:

**No incremental encode.** Every re-run of the open segment uploads the whole
30 s window and recomputes the mel and the encoder — about 10 ms — for 0.5 s of
new audio, which is 24 full encodes for one jfk-length utterance. A `Whisper`
that held an encoded window and only re-decoded would roughly halve a live run.

**No `<|startofprev|>` prompt.** `Whisper::prompt()` is fixed at
`<|startoftranscript|><|notimestamps|>`, so nothing can carry the previous
segment's text across a cut. Measured on jfk.wav repeated three times, the
segment after the 12 s cut loses the "And so" that preceded it.

**No timestamps out.** The timestamp ids are unmasked and nothing surfaces them,
so a segment boundary can only sit where the block classifier guessed rather
than at a decoded silence. The two together are the fix for the row above.

**No token callback.** `transcribe()` returns when it is done; the decode loop
reads a token per step as it goes (see the second performance round below), so
the hook a partial-result callback would hang off exists internally.

The policy's own numbers, measured rather than picked: the longest internal
silent run in jfk.wav at -40 dBFS is 9 blocks — 0.9 s, the pause after "can do
for you" — so the 1.0 s hold clears it by a single block, and either a 0.8 s
hold or a -35 dB threshold splits that sentence in two. A live run over the
whole recording is 24 model runs over 130 chunks, 0.717 s of wall clock in a
Debug build with the last run at 0.038 s, and the committed line comes out
byte-identical to the pinned fixture.

## The second performance round

Where the first round rewrote kernels, this one measured what was left and
then filled the eacp gaps it named, to see which of them were worth the
filling. The runtime stood at 0.033 s on jfk.wav against whisper.cpp Metal's
0.036 s; the decoder was the gap — 22 ms against 15, 0.9 ms a step against
0.6 — and the encoder was within 25% of it.

### Where the time went

A throwaway profiler (the session's `KernelProfile.cpp`, a target of four
lines that is not in the tree) timed every kernel shape a step and an encode
use, as 200 back-to-back dispatches in one command buffer, and then a replica
of the whole decode step — the real 65 dispatches, distinct weights per layer,
so the pipeline switches and the weight traffic are the real ones.

| decode step kernel | per dispatch | per step |
| --- | --- | --- |
| `SplitLinear` 384 → 51864, the logits | 174 us | 174 |
| `SingleQueryAttention` over 1500 keys | 22 | ×4 = 89 |
| `SplitLinear` 384 → 384 | 3.3 | ×24 = 80 |
| `SingleQueryAttention` over 27 keys | 12.7 | ×4 = 51 |
| `LayerNorm` [1, 384] | 3.9 | ×13 = 51 |
| `SplitLinear` 1536 → 384 | 8.3 | ×4 = 33 |
| `SplitLinear` 384 → 1536 | 3.5 | ×4 = 14 |
| `Argmax`, `Embed` | 5.4, 1.2 | 7 |
| **sum of the parts** | | **493 us** |
| **the step as recorded** | | **697 us** |

The parts sum to 493 us and the chain costs 697: **the marginal cost of a
dispatch in the real chain is about 5.5 us**, against 1.7 us for the same
trivial kernel dispatched back to back, and 65 of them are over half the
step. Two ablations of the replica agree — fusing q/k/v took 8 dispatches
and 45 us out, dropping the layernorms took 13 and 71 us — and the key count
of the self-attention does not matter at all (27 keys 697 us, 200 keys 702),
so that kernel is pure latency. An F16 token embedding took the logits row
from 174 to 90 us, 64 us off the step.

The encoder's 8.6 ms of layer kernels: attention apply 432 us, scores 346,
softmax 207, fc2 404, fc1 321, the four 384-wide projections 104 each, over
four layers; the products run at 4.0 to 5.5 TFLOPS against about 14 of FP32
peak. Two things that do not help it, measured: F16 weights in the encoder
products (105/330/404 us against 104/320/405 — they are not weight-bandwidth
bound), and any wider block per thread at a 64-thread group (8×4, 4×8 and 8×8
are all slower than 4×4; 8×8 twice as slow).

### The eacp round

Branch `whisper-performance-gaps` in eacp, from `develop` at `03893c1`, four
changes each written by an agent in a worktree of its own and folded onto the
branch — 37 files, the whole tree building and 1761 tests passing. WhisperEACP
builds against it with `-DCPM_eacp_SOURCE=$HOME/Code/eacp` until it is on
`develop`.

| change | what it is | what eacp measured |
| --- | --- | --- |
| `CommandBuffer::submit()`, `wait()`, `isComplete()`, `read()` | a submission with nothing to settle, a wait scoped to that command buffer rather than to the newest one, and a read that trusts it; `GpuTimestamps::maxTimedPasses` 16 → 128 | a 100-step dependency chain of one small kernel, `commit` and read each step against record-ahead, `wait(k-1)`, `read(k-1)`: **0.100 → 0.044 ms a step** |
| `DispatchOrder::Concurrent` on `beginCompute`, `ComputePass::barrier()` | `MTLDispatchTypeConcurrent`, a null UAV barrier on D3D12, a compute-to-compute pipeline barrier on Vulkan; the serial pass now promises its dispatches are ordered | 64 tiny dispatches: serial 2.80 us each, concurrent with no barrier 1.82, **concurrent with a barrier after every dispatch 3.46 — worse than serial** |
| `groupSum`, `groupMax`, `groupMin` in the EDSL | `simd_*` plus a cross-simdgroup fold on Metal, an emitted shared-memory tree on HLSL (FXC at `cs_5_0` has no wave ops) and GLSL | a layernorm-shaped kernel over one 384-float row: 3.2 → 2.5 us |
| `ComputeProgram({256})`, `({16, 16})` — a per-program threadgroup shape | flows builder → graph → `ShaderSource` → pipeline → dispatch, the stock 64 / 8×8 / 4×4×4 unchanged for a kernel that names none | a group-per-row sum over 1500-float rows: 256 lanes **1.74× slower** than 64 at 1500 rows (bandwidth-bound already), ~1.4× faster under ~100 rows |

The two backends that cannot run here are written by symmetry against each
file's existing calls and are unbuilt; the first Windows and Linux CI runs
are what checks them.

### What each bought the runtime

Each adaptation was an agent in a WhisperEACP worktree built against the
branch, measured on the benchmark before and after, and kept only if it won.

**The pipelined loop** (`Whisper::transcribe`): one step per command buffer,
`stepsInFlight = 2`, each recorded and submitted before the token of the step
before it is read with `CommandBuffer::read`, so the loop stops at
`<|endoftext|>` with one step in the air rather than up to three computed
past it, and the GPU never idles across a commit. **0.033 → 0.029 s; decode
22 → 19 ms; 0.9 → 0.7 ms a step.** Three and four in flight measured no
better. Recording the prompt step behind the encoder's submit rather than
after its wait was tried and not kept: recording a step is 0.046 ms of host
time, so the overlap is worth at most that.

**The reductions** (`Reduce.h` and everything on it): every hand fold became
the intrinsic, and the lane count became a constructor argument measured per
kernel at the shapes it is dispatched at.

| kernel | tree at 64 | `group*` at 64 | chosen | at the chosen count |
| --- | --- | --- | --- | --- |
| `LayerNorm` [1, 384], the decoder's | 4.56 us | 3.81 | 192 | 3.14 |
| `LayerNorm` [1500, 384], the encoder's | 9.98 | 9.15 | 64 | 9.15 |
| `Softmax` [9000, 1500] | 206 | 205 | 256 | 176 |
| `Softmax` [12, 1500], the prompt's | 10.7 | 9.6 | 256 | 5.1 |
| `Argmax` over 51864 | 6.36 | 5.31 | 64 | 5.31 |
| `SingleQueryAttention`, 1500 / 27 keys | 23.4 / 13.9 | 22.7 / 13.1 | 64 | 128 lanes needs 34 KB of shared memory and Metal refuses the pipeline at 32 |
| `MaxReduceKernel`, 240000 cells | 17.0 | 15.6 | 512 | 10.5 — two rounds instead of three |

**0.033 → 0.031 s; 0.9 → 0.8 ms a step**; the encoder's 135 us of wins are
under the benchmark's resolution.

**The tiled product at a wider group: nothing.** `TiledMatMul` at 16×16 with
a 64×64 tile and the same 4×4 block is 5 to 23% slower on every encoder
shape (fc1 334 → 404 us, scores 365 → 466); 16×8 and 8×16 are within noise of
8×8; every rectangular group is 30 to 40% worse; a deeper slab wins fc2 and
loses conv1, the STFT and the scores. A control instantiation of the
parameterised kernel at 8×8 reproduced the shipping numbers, so the sweep
measured the shape. The reason: a bigger tile cuts global traffic, and the
product is served from cache already (fc1 moves 230 MB in 334 us); the
shared-read-to-FMA ratio is set by the register block, which the group shape
cannot change; and a 64×64 tile barriers eight simdgroups instead of two,
twice a slab. Left in the tree as it was. Two things the sweep found on the
way, unshipped: the A slab stored transposed so a thread's four rows are one
`float4` read is 2 to 5% on nearly every shape, and the store fold evaluates
both sides of its `select`s — `exactGelu` for all sixteen outputs with `gelu`
off, and a global read of `output` with `residual` off, 54 MB of dead reads
on the scores product — which an `ifThen` on the uniform is worth 7% on conv2.

**The concurrent pass, which was expected to lose and did not.** The
recording code now declares its own ordering — a `pass.barrier()` at every
boundary where a dispatch reads what the one before it wrote, in the mel
front-end, both halves of the model and the two-stage kernels, a no-op in a
serial pass — and both passes open `DispatchOrder::Concurrent`, so the only
dispatches left to overlap are a layer's three self-attention projections
and `beginSequence`'s eight cross projections. Measured at four decimals over
eleven interleaved rounds against a serial build: the encode pass concurrent
is 3% off the encode, the decode pass concurrent 3% off the decode and the
step (0.861 → 0.839 ms) — the eight boundaries a step loses pay for the
fifty-seven barriers that remain — and giving the eight cross projections a
concurrent pass of their own changes nothing, since a single 1500-row
projection already fills the device. Together **about 2% of the run.** The
per-layer tests open serial passes of their own, so a missing barrier would
show as a wrong transcript rather than as a layer mismatch that bisects.

**Together: 0.033 → 0.026 s** on jfk.wav, whisper.cpp Metal at 0.037 in the
same runs; encode 0.010 s, decode 17 ms, 0.7 ms a step, the transcript
unchanged and 264 tests green.

### What is still on the table

In this tree, from the profile above, in order of measured value: the F16
token embedding (64 us a step, and 80 MB → 40 MB resident), the layernorms
folded into the projections they feed (71 us), a joined q/k/v tensor from the
loader (45 us), a one-dispatch self-attention for a short cache (the
two-stage form is 13 us of latency for 27 keys), and the two `TiledMatMul`
findings above. In eacp, what the round did not attempt: `simdgroup_matrix`
and its cousins, which is how ggml reaches the throughput this product cannot
at 4 to 5 TFLOPS, and a movable `CommandBuffer`, so a pipelined loop's ring
need not be a row of `std::optional`s.

The third round below took two of those five, left two on the strength of a
measurement that says they are now worth nothing, and found the largest win of
the three somewhere the list did not mention.

## The third performance round

This one is on an **Apple M5 Max**, where the two rounds above were on an M4
Max, so none of their absolute numbers carries over and the first thing this
round did was re-measure the baseline. The order of what is worth doing changed
with the machine, which is the round's main lesson.

| | WhisperEACP before | WhisperEACP after | whisper.cpp Metal | whisper.cpp CPU |
| --- | --- | --- | --- | --- |
| transcribe, median | 0.0213 s | **0.0194 s** | 0.031 s | 0.104 s |
| x real time, over the 30 s window | 1412 | **1550** | 964 | 288 |
| encode, median | 0.00627 s | 0.00620 s | 0.005 s | 0.077 s |
| decode, median, 25 steps | 0.0148 s | **0.0130 s** | 0.0143 s | 0.0147 s |
| decode per step | 0.593 ms | **0.520 ms** | 0.572 ms | 0.586 ms |

**8.8% off the run, 12.2% off the decode**, the transcript unchanged token for
token and 269 tests green. The decode is now faster than whisper.cpp's on the
same GPU rather than 3.5% behind it, and the whole run is 1.6x its.

### How it was measured, which mattered more than usual

The wins here are 0.1 to 1.5 ms on a 20 ms run, and two things had to be fixed
before any of them could be read.

The table prints three decimals, which cannot resolve 0.1 ms — but the
`x real time` row is the window over the median at one decimal, so 1552.4 is
the wall clock to five significant digits. That is the number every comparison
below was actually read from. (The benchmark's own decimals were widened to
five while the round ran and put back afterwards; nothing in the tree changed.)

And a sequential A/B is biased: whichever binary runs first pays for a cold
page cache and a GPU still ramping its clocks. Every comparison here is
therefore **A B B A** per round, four rounds of 30 timed runs each, so the
order effect cancels. Warm run-to-run spread within one binary is about
±0.03%, and the paired difference is what is quoted.

### What changed

| change | what it is | effect |
| --- | --- | --- |
| an fp16 copy of `embed_tokens` for the logits projection | the vocabulary projection reads 40 MB a step instead of 80. `SafeTensors::makeExactHalfBuffer` narrows a tensor and keeps the result **only if every value round-trips**, so it is not a numerical trade; `DecoderWeights::LogitsWeight` asks for the copy and gets nothing when the file would lose something | **−1.52 ms**, the whole run's 7.2% |
| `SingleQueryAttention` at 16 chunks a head | eight chunks over six heads is 48 groups, which does not fill this device. Only the cross attention over 1500 encoder rows still chunks at all | **−0.38 ms** |
| a one-dispatch self attention for a short cache | a cache that fits in one chunk has no maximum to rescale and no sum to join, so the partial normalises its own row and the combine is not dispatched. A step's self attention is 2 to 26 keys on this recording, so this is every one of them; a sequence that runs past 256 tokens chunks again from there | **−0.09 ms**, 8 of 8 runs under the baseline median |

The first is the round's whole story and it rests on a fact about the model
rather than on a kernel.

**tiny.en's F32 weights are fp16 values in an F32 container.** OpenAI ships
Whisper's checkpoints in fp16; HuggingFace's conversion widens them, and
widening is exact. All 167 tensors of `model.safetensors` — 37,760,256 values —
narrow back to fp16 and widen again to the bits they started with, which
`Tests/Model` now asserts over the whole file against a rule written in the
test. So the packed vocabulary matrix is *the same matrix*, the same kernel
sums the same products in the same order, and only the load width differs:
`Tests/Decoder` asserts the logits are **bit identical**, not close, and the
oracle's residual against whisper.cpp does not move in its sixth significant
digit. That is why the switch defaults on. A repo genuinely trained and saved
in fp32 gets no copy and the tied weight, silently and by construction.

What it costs is 40 MB of device memory beside the float table, which stays
because the embedding gather has no packed read — the reason `DecoderWeights`
has always insisted `embed_tokens` be F32.

Three assertions hold the claim up, in the three tiers CLAUDE.md asks for:
`Model/SafeTensors/exactHalfBuffer*` pins the predicate on hand-written files
either side of the line, `Model/TinyEn/weightsAreExactlyHalves` pins the fact
about the real download, and `Decoder/TinyEn/packedLogitsWeightIsIdentical`
pins the consequence as an equality rather than a tolerance. The narrowing
itself was checked against ARM's own `__fp16` over 20 million random bit
patterns and every representable half before it was committed to.

`Kernels/singleQueryAttentionMatchesTheChain` grew two shapes on the far side
of `singleChunkKeyLimit`, because the three it had were all under it and the
one-dispatch path would otherwise have taken the kernel test's whole coverage
with it. Both forms are mutation-checked: forcing every key count through one
chunk fails the new shapes, and forcing every one through the chunked form
still passes.

The dispatch that this halves was measured before it was written, by recording
it twice into the same buffer: the second one writes what the first did, so the
transcript cannot move and the delta is the dispatch's own cost. **133 us of a
593 us step, 23% of it**, moving 79.6 MB at about 600 GB/s — at the machine's
bandwidth, so the only lever was fewer bytes.

### The negative results

**A dispatch costs two different things here, and that decided three of these.**
Priced by duplicating all thirteen of a step's layernorms: chained behind a
`pass.barrier()` they cost 0.65 ms over 25 steps, **2.0 us a chained dispatch**;
left independent in the concurrent pass they cost 0.03 ms, **0.09 us**. The
concurrent pass is doing exactly what the second round added it for, and the
consequence is that plan.md's remaining dispatch-count candidates are worth far
less than the M4's 5.5 us a dispatch made them look.

| tried, or priced and not tried | result |
| --- | --- |
| `ifThen` instead of `select` on the store fold of `TiledMatMul` and `SplitLinear` | **0.4% slower**, consistently. The reason is that the second round's premise was wrong: eacp emits `select` as a **C ternary**, and a ternary short-circuits, so `exactGelu` and the `output[at]` load were never evaluated on the untaken side. The "54 MB of dead reads on the scores product" is true of the EDSL graph and false of the emitted MSL — the emitted line is `(u3 != 0u) ? ((0.5 * t4) * (1.0 + eacpErf(...))) : t4`. Reverted; the record above is corrected here |
| a joined q/k/v tensor from the loader | **not attempted, and it should not be.** The three projections are already independent — no barrier between them, in a concurrent pass — so the eight dispatches it removes cost 0.09 us each. The second round valued this at 45 us a step on a machine whose pass was serial |
| the layernorms folded into the projections they feed | **not attempted.** Thirteen chained dispatches is 0.65 ms, which is the ceiling; the fold would put two group reductions into every one of the 48 groups a 384-wide projection dispatches, where there is one group doing them now, and LN1 feeds three projections and would be computed three times. The arithmetic does not clear the ceiling |
| 32 chunks a head | identical to 16, so the count stops where the device does |

### What this round surfaced

Nothing that needs a change in eacp — the two things it turned up are facts
about what eacp already does, and both correct something written above.

| finding | where, and what it means |
| --- | --- |
| `select` is a ternary, and ternaries short-circuit | `ShaderGraph::addSelect` emits `c ? a : b`, so neither side's loads or calls are evaluated unless taken. The second round recorded the opposite and costed an optimisation on it. Nothing in eacp is wrong; the note about it was |
| the concurrent pass is worth an order of magnitude, not 2% | 0.09 us for an independent dispatch against 2.0 us for a chained one. The second round measured the concurrent pass at "about 2% of the run" because only eight of a step's sixty-five boundaries were unordered. The lever it gives is much larger than that number suggests, and the way to use it is to *create* independent dispatches rather than to remove dispatches |

### What is still on the table after this

The step is 520 us and the encode 6.2 ms. The vocabulary projection is now
about 70 us of a step and still the largest single dispatch in it; there is no
third halving of it. What is left, in the order this machine values it:

- **The cross attention's KV cache in fp16.** 18.4 MB read a step, half of it
  saved. Unlike the weight above this one is *not* free: the cache is projected
  from the encoder's rows at run time, so its values are not fp16 to begin with
  and storing them narrowed would be a real approximation.
- **`simdgroup_matrix` in eacp**, still — the tiled product's 4 to 5 TFLOPS
  against ggml's, and now the encoder is 32% of the run.
- **The two unshipped `TiledMatMul` findings**, of which one is now known to be
  a non-finding; the transposed A slab was not retried this round.
- **A movable `CommandBuffer`** in eacp, so the pipelined loop's ring need not
  be a row of `std::optional`s.

## Fourth performance round: simdgroup matrices

The item the last three rounds kept deferring. eacp grew a SIMD-group matrix —
`SimdMatrix`, `simdGroupIndex()`, `multiplyAccumulate()`, `write()` — and every
product in the tree was rewritten out of it and switched over one role at a
time. Same M5 Max, same jfk.wav, same 30 timed runs.

| | WhisperEACP before | WhisperEACP after | whisper.cpp Metal | whisper.cpp CPU |
| --- | --- | --- | --- | --- |
| transcribe, median | 0.01868 s | **0.01770 s** | 0.031 s | 0.103 s |
| x real time, over the 30 s window | 1606 | **1696** | 976 | 292 |
| encode, median | 0.00612 s | **0.00523 s** | 0.005 s | 0.076 s |
| decode, median, 25 steps | 0.01238 s | **0.01229 s** | 0.0143 s | 0.0140 s |

**5.2% off the run and 14.6% off the encode**, the transcript unchanged token
for token against both whisper.cpp columns, and 284 tests green. The encode is
now within 5% of whisper.cpp's own — the row this round was aimed at — and the
whole run is 1.74x its.

### The kernel

`Kernels/SimdTiledMatMul.h`, beside `TiledMatMul.h` and templated over the same
`OperandLayout` and `WeightStorage`, computing the same `TiledMatMulShape` with
the same batch and stride semantics and the same fold on the store. A role is
switched by an alias and no call site moves.

256 threads is eight SIMD groups; they stand two deep and four across over a
64 x 64 tile of C, so each holds 32 rows by 16 columns of it as eight 8 x 8
accumulator fragments. The inner dimension goes by in slabs of 32, staged into
one `shared<Float>(64 * 32 + 32 * 64)` array, and the tile of C goes back out
through that same array so the copy-out can be guarded element by element — a
fragment is loaded and stored whole and has no per-element guard to put on one.
The store's fold happens in that copy-out, which every element passes through
anyway.

Four numbers decided the shape, all of them measured in eacp's own
`Tests/GPU/SimdMatrixTests.cpp` before any of this was written: eight
accumulator fragments a SIMD group rather than sixteen (sixteen collapses to
under 1 TFLOPS), staging C back through threadgroup memory rather than straight
to the buffer (free), storing the fragments unconditionally rather than
branching around them (branching halves the throughput), and no bank-conflict
pitch on the A slab, `simdgroup_load` wanting the natural stride.

### What each role bought, and which were kept

Each switched on its own and measured A B B A against the build before it, four
runs of 30 a side, read from the `encode, median` row at five decimals — the
benchmark's decimals were widened while the round ran and put back afterwards,
as in the third round.

| role | shape it dispatches | effect | kept |
| --- | --- | --- | --- |
| the encoder's linear projections — q, k, v, out_proj, fc1, fc2, and both convolutions' unfold products | [1500, 384] x [384, 384] and x [384, 1536], plus [3000, 240] x [240, 384] and [1500, 1152] x [1152, 384] | **−0.57 ms of encode**, −9.3% | yes |
| the encoder's attention apply | [1500, 1500] x [1500, 64] per head, B read along n | **−0.19 ms**, −3.4% | yes |
| the encoder's attention scores | [1500, 64] x [64, 1500] per head, six batches | **−0.13 ms**, −2.3% | yes |
| the decoder's `beginSequence` cross projections | [1500, 384] x [384, 384], eight of them once a sequence | **−0.13 ms of decode**, −1.1% | yes |
| the mel front-end's STFT product | [3000, 400] x [400, 402] | **−0.02 ms**, at the edge of what the harness resolves | yes |

All five, then — the first time in four rounds that nothing had to be handed
back. What separates them is only size: the encoder's linears are 78% of the
round's win, and the STFT is a rounding error that was kept because every
reading of it fell on the same side.

The attention scores were split onto a program member of their own to be
measured at all: they had been dispatched through the encoder's `projection`,
a key row being contiguous along the head dimension exactly as a weight row is
along its inputs, so switching the projections switched them silently. They
stay split, because a [1500, 1500] product over a head's 64 columns is not the
shape a projection is and the two roles should be free to disagree.

### The negative results

| tried, or found and not taken | result |
| --- | --- |
| zero-filling both slabs past the inner extent | **half of it is dead.** Dropping the zero-fill on either operand alone changes no answer and fails no test, because a term is zero if *either* factor is: the partial slab's A is already zeroed where B's tail is a clamped weight, and the other way round. Both were kept anyway — one of them is load-bearing, neither is obviously the one, and correctness resting on `0 * x` is not worth two `select`s. It is recorded here because a mutation check that drops one and passes looks like missing coverage and is not |
| the register-tiled kernel, retired | **not retired, and it should not be.** eacp lowers a fragment to 64 floats of each thread's own wherever there is no wave matrix instruction, which is correct and does the arithmetic 32 times over. `TiledMatMul` stays as the Windows and Linux fast path, and the role aliases in `SimdTiledMatMul.h` are what pick between them |
| one tolerance moved | `Kernels/simdTiledLinearEncoderWidths` is the only check in the tree that asks a product of 1536 terms, and 1e-5 relative to the answer is the wrong measure for one: the partial sums reach ±80 on the way to an answer that cancels to 0.1, so the error is the accumulation's and not the answer's. `TiledProduct::dotProductTolerance` adds one float epsilon per term and that test alone passes it. **No existing tolerance moved** — every other check still asks 1e-5, and the encoder's double-precision references and the oracle's residuals did not move at all |

### What this needed in eacp

Nothing. The primitive had already landed on the `simdgroup-matrix` branch, and
this round did not have to add an overload to it or fix its lowering — which is
the first time a round here has used something new in eacp without immediately
finding a gap beside it. The four rules the README names (a fragment loaded and
stored whole, offsets and strides uniform across the group, every thread
reaching every operation, a threadgroup that is a whole number of SIMD groups)
were enough to write a batched, strided, folded, ragged-edged product against.

**The tree therefore needs eacp's `simdgroup-matrix` branch until it lands on
develop** — but it does not *require* it. `CMake/Findeacp.cmake` asks the eacp
it was handed whether `ShaderBuilder.h` declares `simdMatrix` and defines
`WHISPER_EACP_HAS_SIMD_MATRIX` from the answer; without it the header declares
no program, the role aliases all name the register-tiled product, and the tree
builds and transcribes exactly as it did before this round, saying so at
configure time. A fetch at develop is 269 tests green; the branch is 284, the
fifteen extra being the new kernel's own.

### What is still on the table after this

The encode is 5.2 ms and the step 492 us. The encoder is now 30% of the run and
the products inside it are no longer the largest thing in it.

- **The cross attention's KV cache in fp16**, still, and still not free — see
  the third round.
- **The 64 x 64 tile against the decoder's few rows.** `beginSequence` was the
  only decoder product worth switching because everything else a step does has
  one to sixteen rows, where a 64-row tile is 98% waste and `SplitLinear` is
  the right kernel. A SIMD-group form of *that* — the inner sum split across a
  group, out of fragments — is a different kernel, not this one re-tiled.
- **A shape that is not a multiple of 64 pays for it.** The encoder's 1500 rows
  leave 36 of the last tile outside the shape on every one of its products,
  which is 2.3% of the arithmetic thrown away. The register-tiled kernel's 32
  wastes 0.7%. Nothing was done about it and nothing obvious can be.
- **A movable `CommandBuffer`** in eacp, unchanged from the third round.

## Fifth performance round: the live loop

The user's complaint was not a benchmark row. Running `Apps/Demo/LiveTranscribe`,
the runtime "takes about 20-30% of GPU on a pretty high end device". This round
is on an **Apple M4 Max** again, so the third and fourth rounds' M5 numbers do
not carry over, and the first thing it did was find out what that 20-30% was
measuring.

### The number was the counter's, not the model's

A throwaway driver streamed jfk.wav plus a 1.5 s gap through `LiveTranscriber`
at the demo's 33 ms tick and added up what `Whisper::transcribe` cost: **53
runs in 28 s, 1.36 s of model time, 4.9% of the wall clock**, 25.7 ms a run.
Continuous speech, 25 s segments, was 6.4%. Meanwhile macOS's GPU utilisation
(`ioreg -r -c AGXAccelerator` "Device Utilization %", which is what Activity
Monitor draws on) read 50-80% at one sample a second — and `68 0 0 0 0 0 68 0
...` at twenty a second. The counter is a short-window snapshot: a 26 ms burst
every half second reads as the burst, not as the duty. There is no way to
read the true share from outside the process without root, so the runtime
now reports it itself (`Benchmark --live`, below).

Two things the driver did show. A run at live cadence costs **about twice**
what the benchmark measures for the same audio (encode 13.8 ms against 7,
0.86 ms a step against 0.6), because the GPU clocks down between bursts; a run
every 4 s cost 49 ms, its encode 34. And every run encoded the full
zero-padded 30 s window — 1500 positions of encoder and 1500 rows of
cross-attention K/V a step — for a segment that is mostly a few seconds long.

So the round had two halves: do less per second of speech, and make each run
cheaper. Four worktree agents, one per lever, each with its own Release
build and a lock serialising every timing run on the one GPU.

### Where it ended

Same jfk.wav, same 30 timed runs, paired **A B B A** twice against the tree
before the round, read from `x real time`:

| | before | after | whisper.cpp Metal | whisper.cpp CPU |
| --- | --- | --- | --- | --- |
| transcribe, median | 0.0216 s | **0.0176 s** | 0.036 s | 0.100 s |
| x real time, over the 30 s window | 1392 | **1700** | 837 | 300 |
| encode, median | 0.0068 s | **0.0064 s** | 0.008 s | 0.072 s |
| decode, median, 25 steps | 0.0147 s | **0.0111 s** | 0.015 s | 0.015 s |
| decode per step | 0.586 ms | **0.445 ms** | 0.60 ms | 0.60 ms |
| transcribe at 704 positions, `--audio-ctx=audio` | — | **0.013 s**, 2254 x real time | 0.032 s | 0.053 s |

**22% off the run with the same work, 40% off it with the audio context**, the
transcript token-identical to whisper.cpp on both columns at both settings,
and 307 tests green (284 before the round). The whole run is now 2.0x
whisper.cpp's at the window and 2.5x at the reduced context.

And the row that answers the complaint, `Benchmark --live 30 1.5` — jfk.wav on
repeat with a 1.5 s pause, 30 s of stream:

| | before | kernels only | as `LiveTranscribe` now runs it |
| --- | --- | --- | --- |
| runs | 57 | 49 | 49 |
| model time | 1.44 s | 1.25 s | **0.84 s** |
| share of the wall clock | 4.8% | 4.2% | **2.8%** |
| per run, mean | 25.2 ms | 25.5 ms | **17.2 ms** |
| encode / decode, mean | 14.7 / 11.4 ms | 14.2 / 10.5 ms | **6.2 / 10.0 ms** |

**42% less model time per second of speech**, the committed transcript
unchanged. The demo app is the one call site that turns the audio context on.
Speech with no pause in it, `--live 30 0`, is the other end: segments run to
the 25 s cut, so the context is close to the window and the option buys
less — 5.5% to 4.5%, 30.9 to 25.2 ms a run at 24 steps.

### The four levers

**1. Encode only the audio there is** — whisper.cpp's `audio_ctx`, as
`Whisper::setAudioContext(positions)` with `audioContextForSamples(count,
margin)` to size it, `LiveOptions::encodeOnlyTheAudioThereIs` to let the live
loop use the segment's length, and `--audio-ctx=N|audio` on `Transcribe` and
`Benchmark`. Default off, and off is byte-for-byte what the tree did before:
A/B against the tree before the option was +0.6%, inside the spread.

Nothing is allocated per run. `Encoder::encode` and `Decoder::beginSequence`
take a position count and dispatch over a prefix of the window's buffers:
conv1's unfold bound (the mel's stride stays 3000), both convolutions' rows,
the positional rows used, every layer norm, projection and attention product,
the score buffer as a compact `[heads, P, P]`, the cross K/V rows the
projections write and every step's cross attention reads. The mel still runs
over the whole window, deliberately: `logMel` is band-major so a frame prefix
is strided, and the peak Whisper clamps against is a global maximum, which is
what whisper.cpp also computes over the full window whatever `audio_ctx` is —
truncating ours would give up the token-for-token agreement below exactly
where a caller sets a context shorter than the audio.

Quality was measured, both sides at the same context:

| positions | jfk.wav, 11 s = 550 positions | whisper.cpp at the same `audio_ctx` |
| --- | --- | --- |
| 1500, 1024, 768, 704, 640 | the window's transcript, 24 tokens | identical |
| 576 | the comma after "for you" is gone, 23 tokens | identical |
| 512 | a full stop and a capital instead of the comma | identical |
| 384 | truncated at 18 steps | — |
| 320 | **runaway**: 446 tokens of "and so and so", 0.25 s | — |

`whisper_full_params.audio_ctx` is the only public way into whisper.cpp's
encoder context and it stays armed on the state, so
`Tests/Oracle/Common.h::armOracleAudioContext` runs one full pass at the
context and every stage call after it runs there. The reduced decoder's
logits track the reference *better* than the full window's at 1024 and 768
(rms 0.037 and 0.062 against 0.465 at 1500, all against a one-ulp resolution
of 0.028) and worse at 576 (rms 0.556), which is where the transcript changes:
that context sits on a decision boundary, and both sides sit on the same side
of it.

The round's real finding is in the last two rows. **A context sized to the
audio plus a second sends the decoder into a repetition loop** — the first
live run of this measured 65.6 ms a run and 103 steps against 13, the encoder
at 4.6 ms and the decoder at its 446-token limit. Two causes, each measured
over prefixes of jfk.wav from three starting points: a context far below the
audio loops (1 s at 128 positions, 2 s at 192, 3 s at 256) and so does one
barely above it (4 s of audio at 256, 9 s at 512), while two tiles clear is
always the window's transcript. Hence `audioContextFloor = 448` and a default
margin of 2.5 s (125 positions, never under two tiles after rounding). The
encoder half of `audio_ctx` is exact and free; the whole risk is what a short
context does to the decoder.

Fourteen tests. The reduced encoder and decoder are checked against *the
shorter shape's own* forward pass with real weights (a reduced run is a shape
built at that length, which is what makes the assertion say something), with
the rows past the context poisoned so a kernel reading the whole buffer cannot
pass; two oracle tests at 576 and 1024/768; and the shared `preparedModel()`
in `Tests/Whisper` means any test that sets a context must put it back, which
`Live/leavesTheContextAloneByDefault` pins.

**2. A run only when there is something new to hear.** `runIsDue()` fired
whenever `stepSeconds` of audio had arrived, speech or not, so the hold before
a segment closes and a pause inside one kept re-running the model on nothing.
`LiveOptions::minNewSpeechSeconds` (0.3 s, the same as `minSpeechSeconds`: the
speech that makes a step worth re-running is the speech that would have made
a segment worth running) gates the periodic run; the closing run is untouched,
since the silence after the last word is what lets Whisper finish it. One
speech block was measured first and removes only 3.5% — a skipped run does not
advance `coveredSamples`, so the next speech block fires it anyway. 0.3 s is
57 → 49 runs and −11% model time at a 1.5 s gap, 48 → 42 at 4 s, transcripts
identical, and 0.5 s would be −21% at the cost of lagging broken-up speech by
a whole step.

**3. The decode step**, profiled here as 61 labelled passes (a floor of
2.7 us each, so the parts sum to 509 us where the real concurrent pass is
445): the logits projection 98 us, the four cross attentions 21 each, the four
self attentions 18, fc2 20 apiece, 24 384-wide `SplitLinear`s at 7.3, thirteen
layernorms at 4.7. A chained dispatch costs **3.14 us** on this machine and an
independent one 0.013 — between the M4's 5.5 of the second round and the M5's
2.0, and the independent one is free everywhere.

| change | effect on the decode |
| --- | --- |
| `SplitLinear`'s split count is the shape's, not a hard-wired 8: `Decoder::stepSplitCount` 96 (a 384-wide row is exactly 96 float4s) and `logitsSplitCount` 32. The count decides how many groups a dispatch has — 48 groups for one row was a few thousand threads on a device sized for tens of thousands — and at 64 lanes or more the fold is `groupSum` | **−18.6%** |
| `SingleQueryAttentionRow`, a kernel of its own for a cache one group takes whole: a lane per output column, one accumulator, no partials array. 18.4 → 7.9 us; the partial's one-chunk branch is gone | **−4.0%** |
| `chunksPerHead` 16 → 24, re-priced here: 8 → 11.56 ms, 16 → 11.37, **24 → 11.07**, 28 and 32 → 12.0, 48 → 12.5. A floor, then a cliff | **−2.6%** |

The split count belongs to the shape and not to the weight: keyed on
`WeightStorage`, `Decoder/TinyEn/packedLogitsWeightIsIdentical` failed at
once, because the float and packed logits summed the same matrix in different
orders — the third round's bit-identity claim doing exactly its job.
`SplitLinear` had no kernel test of its own until now; three were added, on
`TiledProduct::dotProductTolerance` for the 384- and 1536-term shapes.

**4. The encoder's softmax is not a dispatch.** The encode profiled here: the
attention chain 46.6% of 6.6 ms (scores 1.07 ms at 6.4 TFLOPS, softmax 0.69,
apply 1.32 at 5.3), the FFN 26%, the four 384-wide projections 16%, the mel
front-end 4.8%, the convolutions 5.3%. The obvious first cut — a two-sweep
stats kernel writing `[max, 1/sum]` a row, the apply folding both — bought
only 0.13 ms: the softmax pass is bound by the one DRAM read of the 54 MB
score matrix and its other four traversals are cache hits, so the win is not
in a cheaper pass but in no pass. `TiledMatMul` and `SimdTiledMatMul` grew two
template axes: `RowMaxima` has the scores product write, as it stores each
64 x 64 tile, the largest value in each row of the tile (3.5 MB a layer); `AFold`
has the apply fold those maxima into the shift it exponentiates against as it
stages A, sum what it staged — one group's inner loop walks the whole row, so
the sum is exact — and divide the row by the sum on the store. The score
matrix is written once and read once. **−0.41 ms of encode, −6.0%, 6 of 6
pairs**, no tolerance moved, and four tier-1 tests that run both programs into
one command buffer against a scalar scan of the scores the same run produced.
`Softmax` stays for the decoder's prompt step.

### The negative results

| tried, or priced and not tried | result |
| --- | --- |
| the cross-attention KV cache in fp16 — the third round's top remaining candidate | **worth ~2% here, for a real approximation.** Probed by reading the float cache through `readHalf2` (wrong answers, right access pattern, half the bytes): 21.6 → 19.0 us a dispatch. The kernel is not bandwidth-bound — 18.4 MB a step at 213 GB/s on a ~546 GB/s machine — but bound by the shared-memory transpose of its partial rows. Not taken |
| a column-parallel `SingleQueryAttentionPartial`, one kernel for both key counts | 21.3 → 36.0 us: a lane owning a column walks the whole chunk serially. Two kernels it is |
| the partial at 16 or 32 lanes; a hybrid with 1 kB of shared memory | 12.07 / 11.38 ms against 11.04 at 64 lanes; the hybrid's best loses to the register form's best. The transpose through 16 kB beats a serial walk even at the occupancy it costs |
| `logitsSplitCount` above 32 | 64 is 4% slower over the decode, 128 12%: 51864 outputs fill the device at any count and more lanes only lengthen the fold |
| a serial decode pass | 11.23 against 11.12 ms concurrent: under the noise, and the more variable of the two |
| a joined q/k/v, the layernorms folded into their projections | still not attempted: the eight dispatches are independent and cost 0.013 us, and a 384-wide projection now dispatches 384 groups, so a folded layernorm would do its reductions in 384 places instead of one |
| a fused (flash) attention for the encoder out of `SimdMatrix` | **not attempted, for a number.** The chain's arithmetic is 4 x 3.6 GFLOP: 1.7 ms at the 8.6 TFLOPS fc1 reaches, 2.7 at the apply's 5.3. The chain now costs 2.81 ms, so the fused kernel is worth between 0.1 and 1.1 ms, all of it decided by what the interleaved softmax costs in registers — sixteen live fragments a SIMD group is the shape the fourth round measured collapsing, a 32-query tile halves it, and the per-row rescale has to go through threadgroup memory or a product against a diagonal fragment. Recorded here with the shared-memory budget worked out (K staged transposed, 16 kB; S/P 8 kB; Q and V straight from the buffer) for whoever does it |
| the row maxima at one lane per tile row, scanning 64 columns | +0.58 ms on the scores product, as much as the pass it replaced; four lanes a row with sixteen unrolled columns, the stored value held in a `var`, is +0.19 |
| folding the maxima once per row through threadgroup memory instead of each of a row's four staging threads folding for itself | +0.015 ms: two barriers cost more than the redundant cache hits |
| the mel filterbank | left, and the best small target: 0.11 ms at 0.9 TFLOPS from a thread per (frame, mel) with stride-201 reads. It is `forLinear(80, 201, 3000)` with the filterbank as A, but for the `log10(max(x, 1e-10))` and the tiny test shapes; worth about 0.08 ms |
| the mel over a frame prefix | not done, for the reason under lever 1 |

### What this surfaced in eacp

- **Float literals are emitted with `%g`**, six significant digits, so a
  constant does not round-trip: `std::numeric_limits<float>::lowest()` reaches
  the shader as `-3.40282e+38`, a different float. It silently perturbs any
  constant with more than six digits and makes `lowest()` unusable as a
  reduction identity the host then compares against exactly; it cost a
  debugging cycle on the maxima test's fully-outside tiles. `%.9g`, or
  `std::to_chars`.
- **No SIMD-group reduction in the EDSL.** `groupMax`/`groupSum` fold a
  whole threadgroup; a fold across the 32 or 64 lanes that share one row of a
  tile goes through threadgroup memory and two barriers, or is given up on as
  the maxima pass did. `simd_max`/`simd_sum` on Metal, `WaveActiveMax` on
  SM6, subgroup ops on Vulkan. About 0.19 ms of the encode here.
- **Passes are timed, dispatches are not**, and a pass has a ~2.7 us floor.
  A stage profile of a chain recorded as one concurrent pass means
  re-recording it as sixty, which removes the overlap being measured and
  inflates a 445 us step to 509. A per-dispatch timestamp inside a pass, or
  at least the floor reported once so a caller can subtract it.
- **`EACP_SHADER(...)` defines the whole reflection**, so a program whose
  bindings depend on a template parameter writes `reflectMembers` out by hand
  — seventeen `visitor(...)` lines in each product here, to keep the maxima
  buffers off the instantiations that do not read them. A conditional tail on
  the macro would remove it.
- `whisper_full_params.audio_ctx` is the only public way into whisper.cpp's
  encoder context, and it stays armed on the state — not an eacp gap, but the
  thing to know before writing an oracle test at a reduced context.

### What is still on the table after this

The step is 445 us, the encode 6.4 ms at the window and 3.0 at 704 positions.

- **The mel is now the encode** at a reduced context: 1.9 of the 3.0 ms.
  The STFT product is 0.14 ms and fine; the filterbank is the 0.11 ms above;
  the rest is the framing, power, peak and normalise passes over 3000 frames.
- **The logits projection** is 85 us of the step and reads its 39.8 MB of
  fp16 weight at 504 GB/s, the machine's bandwidth; there is no third halving.
- **The cross attention** is 115 us, bound by the partial's transpose, which
  neither fp16 nor the column-parallel form improved.
- **Chained dispatch overhead** is the largest bucket left, ~140 us across
  61 dispatches at 3.14 us, and the only lever this machine rewards is
  making dispatches independent rather than removing them.
- **A run at live cadence still costs twice a warm one.** Nothing here
  changes the clock the GPU idles at; less work per run is the whole answer,
  and the live benchmark is where it shows.
- **The fused encoder attention**, with its budget worked out above.
