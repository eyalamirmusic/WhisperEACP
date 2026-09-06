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

### Local dependency checkouts

eacp and MakeASound are fetched via CPM from GitHub by default. To work against
local checkouts — the normal setup here, since this project drives changes to
eacp's compute layer — pass CPM's per-package source override:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DWHISPER_EACP_UNITY_BUILD=OFF \
      -DCPM_eacp_SOURCE=$HOME/Code/eacp -DCPM_MakeASound_SOURCE=$HOME/Code/makeasound
```

Use `$HOME` (not `~`). CMake does not expand `~`, and shell tilde expansion is
suppressed inside quotes — `-DCPM_eacp_SOURCE="~/Code/eacp"` will silently
configure against a non-existent path and fail later with an error about a
missing `eacp-gpu` target.

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
`Tests/Audio` the way a wire format would be.

The remaining layers — the mel front-end, the model weights, the encoder and
decoder kernels — are not written yet.

### Model format

Weights are loaded from HuggingFace **safetensors** — a JSON header followed by
a raw tensor blob, which is a published format rather than a convention read off
another project's source.

A HF Whisper repo carries everything the runtime needs, in four files:

| file | what it holds |
| --- | --- |
| `model.safetensors` | the weights (151 MB for `tiny.en`) |
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
