# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Rules

Claude must never commit or push without explicit permission from the user in
the current conversation.

## Project Overview

WhisperEACP is a GPU-accelerated Whisper speech-to-text runtime built on
`eacp`'s GPU compute stack, with audio captured through `MakeASound`. Model
layers are authored as `eacp::GPU::ComputeProgram` subclasses — C++ structs
whose bodies are written in eacp's shader EDSL — so a kernel written once emits
MSL on Apple and HLSL on Windows, and Metal and D3D12 both come free.

The second goal is upstream: writing a real neural net against eacp's compute
layer is what surfaces what that layer is still missing. A gap belongs in eacp,
not in a workaround here.

## Build Commands

```bash
# Configure
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DWHISPER_EACP_UNITY_BUILD=OFF

# Build all targets
cmake --build build

# Build a specific target
cmake --build build --target AudioTests
cmake --build build --target DeviceInfo

# Run the tests
ctest --test-dir build --output-on-failure
```

### Build Options

- `WHISPER_EACP_UNITY_BUILD` (default `OFF`): compiles the libraries as CMake
  unity builds. Claude must always configure with
  `-DWHISPER_EACP_UNITY_BUILD=OFF` so per-file compile commands land in
  `compile_commands.json` and LSP tooling returns accurate results.

- `WHISPER_EACP_ENABLE_TESTS` / `WHISPER_EACP_ENABLE_APPS` (default: on when
  top-level): the `Tests/` and `Apps/` trees.

- `WHISPER_EACP_CI_BUILD` (default `OFF`): turns on the unity builds CI uses,
  here and in eacp, MakeASound and Miro.

There is no option for the model. It is fetched at runtime, once per machine,
and a configure downloads nothing of it — see "Fetching the model" below. A
build directory from before that change keeps a stale `WHISPER_EACP_FETCH_MODEL`
entry in its cache, which nothing reads.

- `WHISPER_EACP_IOS_TEAM` (default empty): the Apple development team ID
  `whisper_sign_ios_app` signs an iOS app with under the Xcode generator,
  automatic signing and device family `1,2`. Empty, an iphoneos build is left
  unsigned. On iOS, `Apps/Console` is not added and
  `WHISPER_EACP_ENABLE_TESTS` defaults off, since a phone runs neither.

- `WHISPER_EACP_ENABLE_WHISPER_CPP` (default `OFF`): fetches whisper.cpp and
  its GGML `tiny.en`, and builds `Tests/Oracle` against it.

- `WHISPER_EACP_ENABLE_BENCHMARK` (default `OFF`): the same fetch with
  whisper.cpp's backends on, and the `Benchmark` target. See the benchmark
  section below for what that does to the oracle when both are on.

### Building for an iPhone

`CMakePresets.json` carries `ios-device`: the Xcode generator into
`build-ios/`, `CMAKE_SYSTEM_NAME=iOS`, arm64, and `WHISPER_EACP_IOS_TEAM`
set to the user's personal team, `LK9GL8NWU4`. It is for local phone tests,
not for publishing.

```bash
cmake --preset ios-device
cmake --build --preset ios-device      # LiveTranscribe, Debug, -allowProvisioningUpdates
open build-ios/WhisperEACP.xcodeproj   # run the LiveTranscribe scheme on the phone
```

On iOS `Apps/Demo/LiveTranscribe` starts as it does on a Mac, with everything
on the Neural Engine (`Arguments` in `Main.cpp`, the same on both platforms).
Core ML needs iOS 17;
below it the switch is refused and the app stays on the kernels. The first
launch fetches the 151 MB model into the app's sandbox.

### The benchmark, and which whisper.cpp a tree gets

`Benchmark/` times the runtime against whisper.cpp — on Metal and on the CPU —
in one process, behind `WHISPER_EACP_ENABLE_BENCHMARK`. A benchmark of a Debug
build measures the Debug build, so the numbers come from a Release tree of
their own:

```bash
cmake -G Ninja -B build-release -DCMAKE_BUILD_TYPE=Release \
      -DWHISPER_EACP_ENABLE_BENCHMARK=ON
cmake --build build-release --target Benchmark
./build-release/Benchmark/Benchmark                 # jfk.wav, 10 timed runs
./build-release/Benchmark/Benchmark 30 recording.wav
./build-release/Benchmark/Benchmark 30 --units=all --plan
./build-release/Benchmark/Benchmark --live --audio-ctx=audio
./build-release/Benchmark/Benchmark 20 --coreml-decoder=cpu,cpuAndNeuralEngine
./build-release/Benchmark/Benchmark 500 --only="EACP ML dec cpu"
```

`--coreml-decoder=` adds a column per decoder setting, `EACP ML dec <units>`,
with both halves on Core ML and the decode split into the sequence start and
the step predictions. `--only=` times one column alone and none of
whisper.cpp's, which is how the GPU's utilization for one configuration is
watched from outside (`ioreg -r -c AGXAccelerator`).

Where the build has eacp's Core ML runner and the OS loads its programs, a
fourth column, `WhisperEACP ANE`, runs the runtime with its encoder on Core ML
(`Whisper::setEncoderBackend(EncoderBackend::coreML)`), right after the Metal
column, which stays the transcript reference. Its encode is split into the
mel's own command buffer, the prediction and the seam copies. `--units=` picks
the compute units (`cpuAndNeuralEngine` by default, `all`, `cpuAndGPU`, `cpu`)
and `--plan` prints where Core ML placed every Core ML model of every column
(the mel, the encoder, the decoder's sequence start and step), which under the
engine settings can cost the engine compile again, so it is off by default.
`--live` streams every column of ours, then whisper.cpp's, into one table.

The last of ours, `EACP all ANE`, is `Whisper::setNeuralEngineThroughout()`:
the mel, the encoder and the decoder all on Core ML under `cpuAndNeuralEngine`,
whatever `--units` says, and nothing on the GPU. Its `mel` row is the mel
model's own prediction on the engine.

whisper.cpp is fetched once, in the root `CMakeLists.txt`, through
`CMake/WhisperCpp.cmake`, because its targets (`ggml`, `whisper`) are named
globally and a tree holds one build of it. That build is CPU-only, every
backend off, when only `WHISPER_EACP_ENABLE_WHISPER_CPP` is on — the oracle
wants a reference that computes exactly ggml's own CPU path — and whisper.cpp's
own defaults for the machine, Metal and Accelerate on a Mac, whenever
`WHISPER_EACP_ENABLE_BENCHMARK` is on, since a benchmark against a handicapped
whisper.cpp measures nothing. With both on, the oracle tests run against the
benchmark's build with the GPU off at runtime, which is ggml's CPU backend plus
Accelerate through BLAS; their tolerances are measured against the reference's
own resolution, so they hold, and `loadOracle()` prints which build it loaded.

Two things that build taught the tree: enable Objective-C as well as
Objective-C++, since with only OBJCXX on CMake compiles ggml-metal's `.m` files
as C++; and the fetch clears `_LIBCPP_REMOVE_TRANSITIVE_INCLUDES` around
whisper.cpp's subtree, since ggml's `gguf.cpp` does not compile with it on.
`Benchmark/README.md` says what each row measures and what it does not;
`Benchmark/backends.py` covers the Python backends that cannot be in the
process.

### Dependencies are fetched, not taken from the machine

eacp comes from CPM at **`develop`**, MakeASound and NanoTest at `main`. That is
the default and the only configuration Claude should use: the plain configure
line above is the whole story, and no build here points at a checkout on this
machine.

Claude must **not** pass `-DCPM_eacp_SOURCE` or `-DCPM_MakeASound_SOURCE` unless
the user asks for it in the current conversation. A local tree drags whatever is
uncommitted in it into this build, so an unrelated refactor in progress over
there breaks every target here, with the error surfacing inside the dependency
where it reads as ours.

The override exists for the case it is actually for — changing eacp itself
alongside a change here that needs it — and the build returns to the fetch as
soon as that eacp change is pushed:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DWHISPER_EACP_UNITY_BUILD=OFF \
      -DCPM_eacp_SOURCE=$HOME/Code/eacp
```

Use `$HOME` (not `~`). CMake does not expand `~`, and shell tilde expansion is
suppressed inside quotes — `-DCPM_eacp_SOURCE="~/Code/eacp"` will silently
configure against a non-existent path and fail later with an error about a
missing `eacp-gpu` target.

**The SIMD-group matrix is on eacp's develop, and this tree requires it.**
`Kernels/SimdTiledMatMul.h` is written against eacp's `SimdMatrix`, which
landed on develop as `07ee972b`, so the plain fetch has it. A build directory
configured against an older eacp fails to compile that header; reconfigure so
CPM fetches the current develop.

**The Core ML backend is on eacp's develop, and this tree requires it.**
`CoreMLNet` records into eacp's `eacp-ml-graph` and `CoreMLEncoder` runs on its
`ML::Model`, both of which landed on develop as `b89ea2e5` (merged 2026-09-25),
so the plain fetch has them. A build directory configured against an older eacp
fails to build `whisper-net` and `CoreMLEncoder`; reconfigure so CPM fetches the
current develop.

**The Core ML decoder is on eacp's develop, and this tree requires it.**
`CoreMLDecoder` uses `Graph::argmax` (G8), the run-time-mask
`scaledDotProductAttention` (G9) and `MultiArray::copyRows`, which landed on
develop as `259c3bee` (eacp PR #65), so the plain fetch has them. A build
directory configured against an older eacp fails to build `whisper-net` and
`CoreMLDecoder`; reconfigure so CPM fetches the current develop.

## Architecture

New source files are added directly to the module's CMakeLists.txt under the
appropriate `target_sources(...)` call. Platform-specific sources go inside the
matching `APPLE`/`WIN32` branch.

`Lib/` is the include root, so a header of ours is spelled
`<WhisperEACP/Audio/Format.h>`, beside eacp's `<eacp/...>` and MakeASound's
`<MakeASound/...>`. Everything lives in namespace `WSP`.

### CMake modules must not share a name with a dependency's

`CMAKE_MODULE_PATH` is inherited by every subdirectory, and this project's entry
is appended before eacp's and MakeASound's. A module here named the same as one
of theirs therefore **shadows it**, and their own `include(...)` silently picks
up our file. Hence `CMake/WhisperTargetSetup.cmake` rather than
`TargetSetup.cmake`, and the `whisper_` prefix on every function it defines:
CMake functions are global once defined, so a name declared in two projects
resolves to whichever directory was processed last, which is not something a
call site can see.

Check `eacp/CMake/` and `makeasound/CMake/` before adding a module here.

### Every executable names one company

The root `CMakeLists.txt` sets `EACP_COMPANY_NAME` to `eacp`, the company
`ModelFetch::directory()` and the bundle identifiers already name, and
`whisper_set_default_target_setting` stamps eacp's `AppInfo` into every
executable through `whisper_embed_app_info` (the tests through
`whisper_add_test_executable`, NanoTest's executable with the same call after
it; LiveTranscribe has it from `eacp_set_gui_subsystem`). That is what makes
`eacp::ML::defaultCacheDirectory()` one folder for the whole tree,
`~/Library/Caches/eacp/CoreML`, so the demo, `Transcribe`, the benchmark and
`DeviceInfo --plan` compile each Core ML model once between them, under eacp's
2 GiB least-recently-used budget. `DeviceInfo` prints the directory.

### Dependency order

`find_package(eacp)` runs before `find_package(MakeASound)` in the root
CMakeLists. eacp brings Miro and ea_data_structures along, and CPM keys on the
package name, so MakeASound then finds both populated instead of adding a second
copy at a different tag.

### Library (`Lib/WhisperEACP`)

**Core/** — shared types and the library version. Links `eacp-gpu` publicly
rather than `eacp-core`, since every layer above it holds GPU buffers.

**Audio/** — the audio contract Whisper was trained against (16 kHz mono,
400-point STFT, 160-sample hop, 30 s window) and capture through MakeASound.
Those constants are the model's, not choices; they are pinned by value in
`Tests/Audio` the way a wire format would be. `Audio/Capture` is the microphone
half: a `MakeASound::DeviceManager` opened at Whisper's own rate, the selected
channels mixed to mono on the device callback and handed to `drain()` through a
queue that thread never blocks on, with the peak and RMS of the last block for a
meter and the device and channel choice as `setDevice`/`setChannels`.

**Kernels/** — layernorm, GELU, softmax, the tiled matrix product and the
rest as `ComputeProgram` subclasses, shapes as uniforms so the encoder and
decoder can be written out of them. Row-wise kernels give a group to a row
(`Reduce.h`); products are `TiledMatMul` for many rows and `SplitLinear` for a
step's few; a decode step's attention is `SingleQueryAttention`.
`SimdTiledMatMul.h` is the many-row product again out of eacp's SIMD-group
matrices, and the role aliases at its foot — `LinearProduct`,
`AttentionScoresProduct` and the rest — are what every call site names, so
which of the two kernels a product runs is one decision in one place.

**Mel/** — the front-end: Hann, a reflect-padded STFT, the 80 x 201 filterbank,
log10 and Whisper's clamp-and-scale.

**Model/** — safetensors and the two config files, and `ModelFetch`, which is
where those files come from: four `eacp::OnlineResource`s at a pinned HuggingFace
revision. It is why `whisper-model` links `eacp-network`. **Tokenizer/** —
byte-level BPE and Whisper's special tokens.

**Net/** — the seam: `Net` is the op list the encoder and decoder bodies are
written against once (`recordEncoder`, `recordSequenceStart`,
`recordDecoderStep`), and a backend is what runs it. `KernelNet` dispatches
the kernels above; `CoreMLNet` records the encoder into an `eacp::ML` graph,
the mel an fp16 input enumerated over the eighteen audio contexts and every
weight a blob constant named after its safetensors key. Its second
constructor records a fixed-shape decoder program instead: a cache is a
feature of its own (`cacheFeature`), written out whole by a projection that
fills it (the sequence start's cross keys and values) and read in by a
shorter one, which appends one token's rows after all 448 capacity rows and
puts them out as `appendedRowsFeature` for the caller to write back; which
rows are valid is the `self_mask` and `cross_mask` inputs, the position is
the `position` input, and `CoreMLNetOptions::selection` adds the suppression
input and eacp's argmax, so a step's output is a token. Under
`CoreMLNetOptions::opensSequence` the start and the prompt's step record into
one program: the cross caches it fills feed its own attention, the self caches
begin empty, so the prompt's rows attend causally among themselves under a
constant mask and go out as `appendedRowsFeature`, and the prompt's positions
are rows of the table, with no `position` or `self_mask` input. Three step
ops Core ML keeps on the CPU have engine forms beside them: the token and
the position looked up by a one-hot product (`embedsTokenByProduct`,
`embedsPositionByProduct`, the token as two base-256 fp16 digits since fp16
counts exactly only to 2048), and the argmax out of reductions
(`CoreMLSelection::onTheEngine`, exact, ties to the lowest index, its row
maxima taken down the transposed grid because Core ML's CPU path on an M4 Max
gives `max(row, 0)` across a row).
`CoreMLMel` is the mel front end as a graph, frames major: the samples a hop
to a row, the STFT as one product per row offset of the hop rows against a
Hann-windowed DFT basis, every product with its constant on the right and
nothing transposed (Core ML's CPU path on an M4 Max gets a constant left
operand against a transposed input wrong), the
reflect padding a product against a permutation, and two gains (the samples to
a peak of one, the spectrum to a power peak of 2^15) that put Whisper's eight
decades inside fp16's range, their logs taken back off before the floor and
clamp. `whisper-net` links `eacp-ml-graph`, which builds everywhere.

**Encoder/** and **Decoder/** — HF's two halves out of those kernels, the second
over a KV cache. `CoreMLEncoder` (Apple only, behind `if (TARGET eacp-ml)`) is
the encoder as one Core ML model: recorded through `CoreMLNet`, compiled and
cached by `eacp::ML::Model`, run blocking or through `encodeAsync`, with the mel
copied in and the rows widened out across the seam — or, when the decoder is
on Core ML too, left fp16 in its array (`rowsAt`) for the decoder to copy row
for row, never reaching the GPU. `CoreMLDecoder` (Apple only, the same guard)
is the decoder as two Core ML models recorded from `recordSequenceStart` and
`recordDecoderStep`: the start projects every cross cache out of the window's
encoder rows straight into the arrays the step reads and, in the same
prediction, decodes the prompt (`CoreMLDecoderOptions::promptLength` tokens,
two for tiny.en) and gives back the first sampled token with the prompt's key
and value rows; a step takes one token, the caches and the masks as inputs
and gives back the next token and the eight new key and value rows, which it
writes into the cache arrays at the position. So `beginSequence` takes the
prompt and the first step's suppression and returns a token, and a run makes
as many predictions as the kernels make steps. Its masked attentions are
matmul, an additive mask, softmax and matmul by default: eacp's fused masked
op is correct but measured 2.24 ms a step against 0.68 on the CPU.
`CoreMLEncoderOptions::frontEnd` puts the mel on Core ML too, as a model of its
own in front of the encoder's: samples in, the frames-major mel out, the
context's first rows copied into a frames-major encoder input. Not one fused
model, because Core ML allows one enumerated input and runs a flexible model
with a fixed input beside it on the CPU (the reason is in `CoreMLEncoder.h`).
**Whisper/** — the whole runtime, and the greedy search the
decoder leaves to a layer that can hold the generation config. The encoder and
the decoder each run on the kernels or on Core ML (`EncoderBackend`,
`DecoderBackend`), chosen before `prepare()` or switched at run time
(`switchEncoderBackend[Async]`, `switchDecoderBackend[Async]`), one switch at a
time and none while a run is in flight; the decoder's Core ML units default to
`cpu`, where a step measured fastest, and the suppression masks reach its
steps as additive fp16 rows, so its tokens match the kernels'.
`supportsEncoderBackend(backend, units)` is false, and `encoderBackendRefusal`
says why, where Core ML would put the encoder's enumerated shapes on a CPU
that traps (before macOS 27, under `cpu`, or `cpuAndNeuralEngine` with no
engine), so the setters and the switches throw the `ModelError` before
anything is built; `setCoreMLPlatform` replaces eacp's two predicates for a
test. The decoder's programs are fixed-shape and are never refused for it.
`EncoderBackend::coreMLWithMel` is the encoder with the Core ML mel in front,
and the two-argument decoder switch moves the decoder between its units at run
time. `setNeuralEngineThroughout()` is the preset for all of it on the engine:
`coreMLWithMel`, and the decoder on Core ML under `cpuAndNeuralEngine`, which
builds the step with its position and argmax in the engine forms and its token
still gathered on the CPU, the one op of a run left there on purpose (the
token's product reads the 40 MB table a step, 1.33 ms against 1.07); the
sequence start embeds its prompt by the product, once a run, so nothing of
it is off the engine.
`LiveTranscriber`
sits above it: 16 kHz mono samples in and a growing transcript out, with the
open segment re-run as audio arrives and closed into `committed()` on silence or
on length. Every clock in it is audio time rather than wall time, so a run over
a given recording is deterministic.

The pipeline runs end to end: 30 seconds of audio in, a string out, checked
against a double-precision reference at every layer and against whisper.cpp at
the mel, the logits and the transcript. `plan.md` carries what came in which
order, and the gaps this surfaced in eacp, Miro and ResEmbed.

### Model format

Weights are loaded from HuggingFace **safetensors** — a JSON header followed by
a raw tensor blob, which is a published format rather than a convention read off
another project's source.

A HF Whisper repo carries everything the runtime needs, in four files:

| file | what it holds |
| --- | --- |
| `model.safetensors` | the weights (151 MB for `tiny.en`, F32 — see below) |
| `config.json` | layer counts, widths, head counts |
| `tokenizer.json` | the BPE vocabulary and merges |
| `preprocessor_config.json` | the **mel filterbank**, as an 80 x 201 matrix |

`preprocessor_config.json` is the reason nothing here has to reimplement
librosa's filterbank construction or lift one out of another project: the matrix
is in the file, already built. It also carries `sampling_rate`, `n_fft`,
`hop_length`, `feature_size`, `chunk_length`, `n_samples` and `nb_max_frames`,
which agree value for value with the constants in `Audio/Format.h`. Those
constants stay as constants — a shape the code is compiled against, not
something read at runtime — but the file is what to check them against.

Both claims above are checked in `Tests/Model`, and both hold: `mel_filters` is
in the file at 80 x 201, and every constant agrees.

**`tiny.en` ships F32, not F16** — `"torch_dtype": "float32"`, and all 167
tensors in its `model.safetensors` are `F32` (19104-byte header, 151,041,024-byte
blob). F16 is what the larger repos ship. eacp's EDSL still has no half *type* —
`ValueType` is `Float/Float2/.../UInt/Int/Bool` and `InputBuffer` yields a
`Float` — but it does have half *storage*, `InputBuffer::readHalf`, so a weight
may stay packed and be widened as it is read. `TensorBuffer` names which of the
two a buffer holds and the projections pick the matching program.

**And F32 is the container, not the precision.** OpenAI's checkpoints are fp16
and HuggingFace's conversion only widens them, so every one of those 167 tensors
round-trips through fp16 unchanged — asserted over all 37,760,256 values in
`Tests/Model`. That is why `Whisper::setPacksWeights` is on by default: under
`WeightPacking::ExactHalf` every projection weight of the encoder and the
decoder goes to the device narrowed, and the logits projection reads an fp16
copy of `embed_tokens` beside the float table the gather reads. It halves the
weight bytes a step reads, and the transcript and the logits come out bit
identical.
`SafeTensors::makeExactHalfBuffer` is what enforces the "bit identical" — it
returns nothing when a file would lose something, so a repo genuinely saved in
fp32 keeps the float weights it shipped.

### Fetching the model

At runtime, through **`eacp::OnlineResource`** — one resource per file, fetched
the first time anything asks and found on disk every time after.
`Model/ModelFetch.h` is the whole of it and is the only place the four URLs are
written down:

| member | what it is |
| --- | --- |
| `ModelFetch::directory()` | where the four files live |
| `ModelFetch::resources()` | the four as `OnlineResource::Info`, weights last |
| `ModelFetch::isAvailable()` | all four on disk at this revision, no request made |
| `ModelFetch::fetch(freshness, onProgress)` | blocking, for a console `main` |
| `ModelFetch::Download` | the async form: `start()`, `cancel()`, `progress()`, `onFinished` |
| `ModelFetch::progressText(progress)` | the one sentence every caller that draws a download draws |

**The URLs name a commit, not `main`.** `ModelFetch::revision` is
`87c7102…`, which is `openai/whisper-tiny.en` as every number in `plan.md` was
measured against, and it is the `Info::version` as well, so bumping it
re-downloads instead of trusting the previous revision's copy. That revision is
what replaced the four `URL_HASH` SHA256s the configure used to pin:
`OnlineResource` has no content hash, so the integrity claim is now HTTPS plus
an immutable URL rather than a digest the project checks — `plan.md` says what
that costs.

**`Freshness::trust` everywhere, and that is not laziness.** The bytes at a
commit cannot change, so a revalidation can only waste a round trip — and for
`model.safetensors` it wastes 151 MB: HuggingFace redirects an LFS file to a
CDN, eacp records the ETag of the response it ended on, which is the CDN's
rather than the one the origin compares against, and a conditional GET
therefore misses and re-downloads the whole file. Measured, not assumed, and
written up in `plan.md` as an eacp gap. The three small files revalidate
correctly.

**One directory for every binary here**, and not eacp's default. That default is
`FilePath::appSupportDirectory() / "Resources"`, which names the folder after
the running executable — right for an app that ships, wrong for a tree of
eleven binaries that share one 151 MB model. So `ModelFetch::directory()`
spells the names instead, through the two-argument overload:

```
~/Library/Application Support/eacp/WhisperEACP/Models/tiny.en
```

with eacp's own `<file>.resource.json` sidecar beside each file recording the
URL, the version and the server's validators.

**The main-thread rule is why the fetch is in `main`.** `OnlineResource::fetch`
pumps the event loop until the transfer is done, which is a console `main`'s
privilege and not an event-loop callback's — so every console entry point here
(`Transcribe`, `Benchmark`, the test mains) fetches *before* `eacp::Apps::run`,
and `Apps/Demo/LiveTranscribe`, whose loop is already running, uses
`ModelFetch::Download` instead.

Nothing is copied beside any binary, and nothing is compiled into one.
`Whisper::load(const ModelFiles&)` and `SafeTensors::fromView` stay: they are
the path for a model somebody already holds in memory, and `Tests/Whisper`
exercises them.

### The sample

`Samples/jfk.wav` **is** committed — 352 kB, unlike the model. 11 seconds of
President Kennedy's inaugural address, 20 January 1961, 16 kHz mono PCM16,
byte-identical to whisper.cpp v1.9.3's `samples/jfk.wav`, which is OpenAI
whisper's own `tests/jfk.flac`. A US government work held by the JFK Library
and in the public domain.

`WHISPER_EACP_SAMPLE_DIR` (set in the root `CMakeLists.txt`) points at it. It
stays separate from the model directory because that one is a HuggingFace repo —
the one `ModelFetch` fetched, or a checkout somebody already has — and jfk.wav
is in neither.

### `Apps/Console/Transcribe`

Three forms, and the app says which one it took before printing the transcript:

```bash
./build/Apps/Console/Transcribe/Transcribe                          # fetched model, built-in sample
./build/Apps/Console/Transcribe/Transcribe recording.wav            # fetched model
./build/Apps/Console/Transcribe/Transcribe path/to/model recording.wav
./build/Apps/Console/Transcribe/Transcribe --audio-ctx=audio recording.wav   # encode only the audio there is
./build/Apps/Console/Transcribe/Transcribe --ane recording.wav      # everything on the Neural Engine
```

`--ane` is `Whisper::setNeuralEngineThroughout()`, and the report then splits
the encode into the mel's prediction and the encoder's. The first `--ane` run
on a machine compiles for the engine, about 15 s, into the cache every binary
here shares, `~/Library/Caches/eacp/CoreML`, and every later run of any of them
loads it.

`--audio-ctx=N|audio` is `Whisper::setAudioContext`, whisper.cpp's `audio_ctx`:
the encoder runs over N of its 1500 positions, or over what the recording
needs plus a margin. Off by default everywhere but `Apps/Demo/LiveTranscribe`,
which turns `LiveOptions::encodeOnlyTheAudioThereIs` on; plan.md's fifth
performance round has the transcripts at each context and the floor and margin
that keep the decoder out of a repetition loop.

The first two forms fetch `openai/whisper-tiny.en` before the run loop opens,
printing one rewritten progress line while 151 MB arrives and nothing at all on
every later run. A fetch that fails — no network, a machine behind a proxy —
prints why and returns 2. Arguments are parsed first, so a usage error costs no
download.

### `Apps/Demo/LiveTranscribe`

The runtime over a microphone, in a window: an input device and a channel slice
chosen from two `UI::ComboBox`es filled from MakeASound's
`UIDeviceManager` dropdowns, a level meter, and the transcript as it is spoken.
Three parts under a `UI::ComponentHost`: `MainPanel` is the tree and the layout,
`LevelMeter` draws the last block's RMS as a dB-scaled fill under a peak tick
that holds and falls, and `TranscriptView` wraps the text by hand inside a
`ScrollPanel` — eacp's painter draws one line and measures one, and there is no
wrapped-text call or text-area widget in the tier. `Session` is the non-UI half:
the `Whisper`, the `Capture`, the `LiveTranscriber` over the two, and the
`ModelFetch::Download` that has to finish before any of them exists.

**Everything runs on the message thread**, from one 30 Hz `Threads::Timer`, and
that is a constraint rather than a simplification: eacp's GPU layer is
main-thread only and `Whisper::transcribe` blocks on its own commits, so a tick
drains the capture queue, pushes it, and lets `LiveTranscriber::update()` decide
whether a run is due. The device callback is the only other thread and it
reaches nothing here. `Session::start()` is posted from a `Threads::callAsync`
in the app's constructor, so the window is up before anything slow begins: on a
first run it says `downloading model   model.safetensors (4 of 4)   41.2 of
151.1 MB   27%`, polled off the same timer as the meter, and then "loading
model..." for the half second of mapping and kernel compilation. `ModelState` is
`Downloading`, `Loading`, `Ready`, `SwitchingBackend` or `Failed`, and
`Session::onModelSettled` is what the `--autostart` switch hangs off, since
"after the model loaded" is no longer a place in a function.

The encoder and decoder boxes, on a row of their own under the device and
channel boxes, move each half between the kernels, "GPU (Metal)", and Core ML
at runtime — the encoder to the Neural Engine with the mel on the GPU, "Core
ML", or with the mel on Core ML too, "Core ML + mel"; the decoder to the CPU,
"Core ML (CPU)", or to the engine, "Core ML (engine)", a `DecoderChoice` of
backend and units. The last of each is everything on the Neural Engine, which
is where the demo starts. Two
boxes rather than one of every combination, because the runtime switches the
halves apart and so does `Session`: each box is one request, and a second row
because the first has no room for two more boxes at the window's minimum
width. Each is there only where its `Whisper::supports...Backend` holds, and
each follows the session rather than the click; both grey out while either
half switches, since the session makes one switch at a time, the encoder's
first when both are asked for. The model is always prepared on the kernels,
and where each half starts is only the first switch: "Core ML + mel" and "Core
ML (engine)" by default on every platform (`--ane` names that), both halves
left on the kernels under `--gpu`, the encoder on "Core ML" under `--coreml`
and the decoder on "Core ML (CPU)" under `--coreml-decoder`, each of those
two moving only its own half. `Session`'s constructor takes the two starts, an
`EncoderBackend` and a `DecoderChoice`, and requests whichever is not the
kernels. A refused switch leaves that half on the kernels, so the default is
safe on a Mac with no engine and on iOS below 17.
`Session::requestEncoderBackend` and `requestDecoderChoice` remember the
choice. What follows is the encoder's switch, and the decoder's is the same
through `switchDecoderBackendAsync(backend, units)`. The first tick with no
run in flight begins the switch, and the tick after issues
`Whisper::switchEncoderBackendAsync`, so the status line already says
"compiling encoder for Core ML..." or "switching to GPU..." before anything
blocks. That call re-prepares the encoder alone. To Core ML, the recording is
on the main thread and the compile and load are on the model's queue
(`CoreMLEncoder::prepareAsync` over `ML::Model::loadAsync`). The first compile
on a machine takes about 14 s, and the window stays live throughout. To the
kernels, it is blocking and fast. Until the switch settles, no run starts,
Stop defers its flush, and the audio waits in the queue. A switch that fails
leaves the model `Ready` on the backend it was on, with the reason on the
error line. The new encoder is built beside the old one, so a failed switch
changes nothing.

`LiveTranscriber` runs through `Whisper::transcribeAsync` on either backend.
On the kernels that call resolves before it returns, so a run still starts and
ends inside one tick. With the encoder on Core ML it comes back on a later
tick, with the prediction on the model's queue and the decode on the main
thread, and until then `update()` starts nothing. The decoder is blocking on
either backend: on Core ML it is one prediction a token on the main thread,
about 19 ms for jfk.wav in Release against the kernels' 9. A `Whisper` with a
run or a switch in flight
refuses `transcribe`, `transcribeAsync`, `setAudioContext`, `prepare` and
another switch with a `std::logic_error`.

Two build-side requirements. The bundle's plist needs
`NSMicrophoneUsageDescription`, since macOS blocks a capture app that asks for
the microphone without one, and blocks it by hanging inside CoreAudio on the
thread that asked; `eacp_add_plist_entries` adds it to eacp's template, so
there is no copy of the plist here. And `Apps/CMakeLists.txt` adds the tree
behind `if (TARGET eacp-ui)`, eacp building its widget tier only where it has
a Metal or D3D12 backend.

## Correctness validation

Three tiers, in the order a failure should be diagnosed:

1. **A scalar CPU reference written inside the test itself**, for every kernel —
   layernorm, GELU, softmax, matmul, attention. No dependency, runs on any
   machine, and it is the only tier that catches a backend divergence, since the
   same assertion runs against MSL on Apple and HLSL on Windows. This is the
   bulk of the suite.

2. **whisper.cpp as an in-process oracle** for stage-level equivalence, not yet
   wired up. Its public API is addressable stage by stage, which is what lets a
   mismatch bisect to a layer instead of only reporting a wrong transcript:
   `whisper_pcm_to_mel` for the front-end; `whisper_set_mel` + `whisper_encode`
   fed our own mel for the encoder; `whisper_decode` + `whisper_get_logits` for
   the decoder; `whisper_tokenize` / `whisper_token_to_str` for the tokenizer.

   MIT, CMake, and cheap — 31 targets and about 4.5 s to build. Fetch it with
   `WHISPER_BUILD_TESTS`, `WHISPER_BUILD_EXAMPLES`, `WHISPER_BUILD_SERVER`,
   `GGML_METAL`, `GGML_OPENMP` and `GGML_ACCELERATE` off — and `GGML_BLAS OFF`
   as well, which is not implied by `GGML_ACCELERATE OFF`: without it the
   configure still finds Accelerate and links BLAS into the reference.

   It reads GGML `.bin` models, so a comparison against it spans a format
   conversion. That conversion is therefore a candidate explanation whenever
   numbers disagree, and is worth ruling out first.

   Gate it behind an opt-in option, and skip when no model file is present — the
   same shape as the GPU tests skipping when `Device::shared().isValid()` is
   false. A model is a download (75 MB for tiny.en), never a commit.

3. **Small committed fixtures** for what is cheap and stable — the mel
   filterbank matrix, a handful of tokenizer cases.

### Tests (`Tests`)

NanoTest, one executable per module, and two shared entry points in
`Tests/Support/` rather than NanoTest's own:

- `GpuTestMain.cpp` (`WHISPER_GPU_TEST_MAIN`) runs the suite inside
  `eacp::Apps::run`, which owns the run loop and autorelease pool the Metal
  backend is written against. `GPUTests`, `KernelTests` and `MelTests` take it.
- `ModelTestMain.cpp` (`WHISPER_MODEL_TEST_MAIN`) is that with
  `ModelFetch::fetch` in front, for the six modules that read the real tiny.en
  files — `Model`, `Encoder`, `Decoder`, `Net`, `Whisper`, `Oracle` — plus `Tokenizer`,
  which wants `tokenizer.json` and no device. One or the other, never both: the
  fetch has to happen outside the loop the other one opens.

A module that needs the model therefore has it by the time its first test runs,
and the first executable of a suite run is the one that pays for the download.
Two things follow. A test that needs a file still returns early when it is
absent — no network is the case those skips were always written for — and the
fetch is skipped entirely for `--list-tests`, which is how NanoTest's CMake
discovers test cases after every link: its whole stdout is the list of names, so
a progress line there becomes a test case, and a build would otherwise download
151 MB.

A test that needs a device returns early when `Device::shared().isValid()` is
false, so the suite still passes on a machine with no GPU.

Every kernel gets a test that asserts against a scalar CPU reference computed in
the test itself. That is what catches a backend divergence — the same assertion
runs against MSL on Apple and HLSL on Windows.

`Tests/Net` runs the encoder on Core ML against the kernels' rows under each
compute-unit setting, and `Tests/Whisper` and the oracle run the whole runtime
on it. `Tests/Whisper/CoreMLDecoderTests.cpp` runs the Core ML decoder against
the kernel decoder's logits and tokens under each setting, over the window
and at a context only the cross mask holds to, and the whole runtime with
either or both halves on Core ML, with the step's engine forms as well;
`DecoderBackendTests.cpp` is the decoder's switch. `Tests/Net/CoreMLMelTests.cpp`
holds the Core ML mel to the kernels' cell by cell over speech, full-scale
tones, quiet noise and silence, and `CoreMLEncoderTests.cpp` the encoder run
from samples through it; `Tests/Whisper/RefusalTests.cpp` is that refusal on a stand-in platform;
`Tests/Whisper/NeuralEngineTests.cpp` is the
all-on-the-engine preset, its transcript, where Core ML placed its four models,
and the switches that reach it at run time. Every Core ML model they compile
goes to one fixed cache,
`<temp>/whisper-eacp-tests/CoreML` (`sharedCoreMLCacheDirectory()` in
`Tests/Model/Common.h`, handed over through `Whisper::setCoreMLCacheDirectory`
and `CoreMLEncoderOptions::cacheDirectory`), so the 13.5 s engine compile is
paid once per machine rather than once per test binary, and kept apart from
the apps' `~/Library/Caches/eacp/CoreML`, whose budget a test's programs would
otherwise count against. This repository has no CI of its own; a machine
with no Neural Engine (eacp's macOS CI runner is one) places everything on
the CPU, so `Tests/Net` holds every setting to the CPU's tolerance unless
`EACP_REQUIRE_ANE=1`. The `cpu` setting of the enumerated models has run only
on macOS 27, since before it eacp refuses to load them there; `WHISPER_EACP_SLOW_TESTS=1` adds
`Net/CoreML/errorByDepth`, which compiles one fixed program per depth into a
cache of its own and removes it.

## Code Style

Always use the most modern C++ and RAII practices.
Use auto for variables and whenever possible.
Don't use auto for functions and member functions

Don't use comments unless absolutely needed. Use named functions to make code
self documenting.

Give std::function members a non-null default — a no-op lambda, or one
returning an empty value — so call sites invoke them directly without null
checks.

Enforced via `.clang-format`:
- Allman brace style
- 85 column limit
- 4-space indentation (no tabs)
- Pointer alignment: left (`int* ptr`)
- Break constructor initializers before comma

Always run clang-format for edited code files
