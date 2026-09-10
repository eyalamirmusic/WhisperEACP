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

- `WHISPER_EACP_FETCH_MODEL` (default `ON`): the four `tiny.en` files on disk,
  and — through `whisper_bundle_model(<target>)` — copied beside every binary
  that asks for them, after each link. A default configure therefore downloads
  151 MB. Off, the function is a no-op, `Whisper::hasBundledModel()` answers
  `false` at runtime, and the tests that need a file skip.

  A build directory configured before the switch flipped keeps its cached
  `OFF`: reconfigure with `-U WHISPER_EACP_FETCH_MODEL`, or pass `ON`
  explicitly, to get the default behaviour there.

- `WHISPER_EACP_ENABLE_WHISPER_CPP` (default `OFF`): fetches whisper.cpp and
  its GGML `tiny.en`, and builds `Tests/Oracle` against it.

- `WHISPER_EACP_ENABLE_BENCHMARK` (default `OFF`): the same fetch with
  whisper.cpp's backends on, and the `Benchmark` target. See the benchmark
  section below for what that does to the oracle when both are on.

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
```

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

**Model/** — safetensors and the two config files. **Tokenizer/** — byte-level
BPE and Whisper's special tokens.

**Encoder/** and **Decoder/** — HF's two halves out of those kernels, the second
over a KV cache. **Whisper/** — the whole runtime, and the greedy search the
decoder leaves to a layer that can hold the generation config. `LiveTranscriber`
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
`Tests/Model`. That is why `DecoderWeights::LogitsWeight` defaults to keeping an
fp16 copy of `embed_tokens` for the logits projection: it halves the largest
read a decode step makes and the logits come out bit identical.
`SafeTensors::makeExactHalfBuffer` is what enforces the "bit identical" — it
returns nothing when a file would lose something, so a repo genuinely saved in
fp32 keeps its float weight.

### Fetching the model

Models are fetched with **CPM**, like every other dependency, rather than a
hand-rolled `file(DOWNLOAD)`. CPM forwards unparsed arguments to
`FetchContent_Declare`, so a plain URL works. Each file needs
`DOWNLOAD_NO_EXTRACT YES` — without it FetchContent takes the download for an
archive and fails trying to unpack it — and `DOWNLOAD_ONLY YES`, since there is
no CMakeLists to add:

```cmake
CPMAddPackage(
        NAME whisper-tiny-en-config
        URL https://huggingface.co/openai/whisper-tiny.en/resolve/main/config.json
        DOWNLOAD_NO_EXTRACT YES
        DOWNLOAD_ONLY YES)
```

Pin `URL_HASH` on each once the model choice settles.

Do not use `CPM_SOURCE_CACHE`.

The fetch runs behind `WHISPER_EACP_FETCH_MODEL`, on by default, so an ordinary
configure downloads the four files. `WHISPER_EACP_MODEL_DIR` points at the
directory they are linked into either way, so a test compiles against a real
path and skips on a file's absence rather than on an `#ifdef`.
`WHISPER_EACP_MODEL_FILES` lists the four paths, and is empty when the fetch is
off.

### Bundling the model

The model is **not** compiled into any binary. It used to be, through ResEmbed,
and 151 MB as a decimal brace initializer was a 657 MB `.c`, a 42 s compile and
a 16 GiB peak RSS; `plan.md` keeps the numbers. The build copies it instead.

`whisper_copy_resources(<target> FILES ... [DESTINATION <dir>])`, in
`CMake/WhisperResources.cmake`, is a `POST_BUILD` copy to where the binary can
find files at runtime: `Contents/Resources` of a `MACOSX_BUNDLE` target, and
the executable's own directory otherwise — a Windows build, or a macOS
executable that is not a bundle, which is what the console apps and every test
here are. `copy_if_different`, so a rebuild that changed nothing copies nothing.

`whisper_bundle_model(<target>)`, defined in `Model/CMakeLists.txt` beside the
fetch, applies that to the four fetched files under `DESTINATION WhisperModel`.
It exists whether or not the fetch is on and does nothing when it is off, so a
consumer calls it unconditionally and `Whisper::hasBundledModel()` answers at
runtime.

The runtime half is `WSP::resourcesDirectory()` in
`Whisper/ResourcesDirectory.h`, one source per platform: CoreFoundation's
`CFBundleCopyResourcesDirectoryURL` on Apple, which answers `Contents/Resources`
for a bundle and the executable's directory for anything else, and
`GetModuleFileNameW` on Windows. `Whisper::bundledModelDirectory()` is that plus
`WhisperModel`, `hasBundledModel()` checks the four files are in it, and
`loadBundled()` is `load()` on it, so the weights are mapped as for any
directory. The directory name is spelled in `Model/CMakeLists.txt` and in
`Whisper::bundledModelDirectoryName`, and nowhere else.

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
stays separate from `WHISPER_EACP_MODEL_DIR` because that one may point at a
HuggingFace checkout somebody already has, and jfk.wav is not in one.

### `Apps/Console/Transcribe`

Three forms, and the app says which one it took before printing the transcript:

```bash
./build/Apps/Console/Transcribe/Transcribe                          # built-in model, built-in sample
./build/Apps/Console/Transcribe/Transcribe recording.wav            # built-in model
./build/Apps/Console/Transcribe/Transcribe path/to/model recording.wav
./build/Apps/Console/Transcribe/Transcribe --audio-ctx=audio recording.wav   # encode only the audio there is
```

`--audio-ctx=N|audio` is `Whisper::setAudioContext`, whisper.cpp's `audio_ctx`:
the encoder runs over N of its 1500 positions, or over what the recording
needs plus a margin. Off by default everywhere but `Apps/Demo/LiveTranscribe`,
which turns `LiveOptions::encodeOnlyTheAudioThereIs` on; plan.md's fifth
performance round has the transcripts at each context and the floor and margin
that keep the decoder out of a repetition loop.

A form that needs the bundled model in a build that copied none prints how to
get one and returns 2.

### `Apps/Demo/LiveTranscribe`

The runtime over a microphone, in a window: an input device and a channel slice
chosen from two `UI::ComboBox`es filled from MakeASound's
`UIDeviceManager` dropdowns, a level meter, and the transcript as it is spoken.
Three parts under a `UI::ComponentHost`: `MainPanel` is the tree and the layout,
`LevelMeter` draws the last block's RMS as a dB-scaled fill under a peak tick
that holds and falls, and `TranscriptView` wraps the text by hand inside a
`ScrollPanel` — eacp's painter draws one line and measures one, and there is no
wrapped-text call or text-area widget in the tier. `Session` is the non-UI half:
the `Whisper`, the `Capture` and the `LiveTranscriber` over the two.

**Everything runs on the message thread**, from one 30 Hz `Threads::Timer`, and
that is a constraint rather than a simplification: eacp's GPU layer is
main-thread only and `Whisper::transcribe` blocks on its own commits, so a tick
drains the capture queue, pushes it, and lets `LiveTranscriber::update()` decide
whether a run is due. The device callback is the only other thread and it
reaches nothing here. The model is loaded from a `Threads::callAsync` posted in
the app's constructor, so the window is up saying "loading model..." before the
half second of mapping and kernel compilation starts.

Two build-side requirements. The bundle needs an `Info.plist.in` of its own
carrying `NSMicrophoneUsageDescription` — set **after**
`whisper_set_default_target_setting`, which points every bundle here at eacp's
stock template — since macOS blocks a capture app that asks for the microphone
without one, and blocks it by hanging inside CoreAudio on the thread that asked.
And `Apps/CMakeLists.txt` adds the tree behind `if (TARGET eacp-ui)`, eacp
building its widget tier only where it has a Metal or D3D12 backend.

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

NanoTest, one executable per module. `Tests/GPU` has an entry point of its own
(`TestMain.cpp`): anything touching the GPU runs inside `eacp::Apps::run`, which
owns the run loop and autorelease pool the Metal backend is written against. A
test that needs a device returns early when `Device::shared().isValid()` is
false, so the suite still passes on a machine with no GPU.

Every kernel gets a test that asserts against a scalar CPU reference computed in
the test itself. That is what catches a backend divergence — the same assertion
runs against MSL on Apple and HLSL on Windows.

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
