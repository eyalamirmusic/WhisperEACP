# WhisperEACP

GPU-accelerated Whisper speech-to-text on top of
[eacp](https://github.com/eyalamirmusic/eacp)'s GPU compute stack, with audio
from [MakeASound](https://github.com/eyalamirmusic/makeasound).

Every layer of the model is authored as an `eacp::GPU::ComputeProgram` — a C++
struct whose body is written in eacp's shader EDSL rather than in a shading
language. The EDSL emits MSL on Apple and HLSL on Windows from that one source,
so a kernel written once runs on Metal and D3D12 and cannot drift between them.

```cpp
struct ScaleKernel final : ComputeProgram
{
    ScaleKernel() { compile(); }

    void define() override
    {
        auto i = threadId();
        write(output, i, input[i] * scale);
    }

    Uniform<InputBuffer> input;
    Uniform<OutputBuffer> output;
    Uniform<Float> scale;

    EACP_SHADER(input, output, scale)
};
```

The other half of the point is upstream: writing a real neural net against
eacp's compute layer is what surfaces what that layer is still missing, and
those gaps get closed in eacp rather than worked around here.

The pipeline runs end to end: 30 seconds of audio in, a string out, through the
mel front-end, the encoder, the decoder's KV-cached loop and the greedy search,
checked against a double-precision reference at every layer and against
whisper.cpp at the mel, the logits and the transcript. `plan.md` carries what
came in which order, the four performance rounds, and the gaps this surfaced in
eacp, Miro and ResEmbed — the second half of the point.

### Where it stands

`tiny.en` on jfk.wav, an M5 Max, the median of 30 timed runs, against
whisper.cpp v1.9.3 in the same process (the table `Benchmark` prints; see
[Benchmarking](#benchmarking) for what each row does and does not measure):

| | WhisperEACP | whisper.cpp Metal | whisper.cpp CPU |
| --- | --- | --- | --- |
| transcribe, median | **0.0177 s** | 0.031 s | 0.103 s |
| x real time, over the 30 s window | **1696** | 976 | 292 |
| encode, median | **0.0052 s** | 0.005 s | 0.076 s |
| decode, median, 25 steps | **0.0123 s** | 0.0143 s | 0.0140 s |

The transcript is the same token for token in all three columns.

## Layout

| | |
| --- | --- |
| `Lib/WhisperEACP/Core` | Shared types and the library version. Links `eacp-gpu` |
| `Lib/WhisperEACP/Audio` | Whisper's audio contract, and capture through MakeASound |
| `Lib/WhisperEACP/Kernels` | layernorm, GELU, softmax, conv1d, attention and the matrix products — shapes as uniforms, so the encoder and decoder are written out of them |
| `Lib/WhisperEACP/Mel` | Hann, STFT, the 80 x 201 filterbank, log10 and the scaling |
| `Lib/WhisperEACP/Model` | safetensors weights, `config.json`, the filterbank |
| `Lib/WhisperEACP/Tokenizer` | byte-level BPE and Whisper's special tokens |
| `Lib/WhisperEACP/Encoder` | HF's `WhisperEncoder.forward`, recorded into one command buffer |
| `Lib/WhisperEACP/Decoder` | The same for the text half, over a KV cache |
| `Lib/WhisperEACP/Whisper` | The whole runtime — samples in, a transcript out — and `LiveTranscriber` above it |
| `Apps/Console/DeviceInfo` | What this machine offers: GPU limits and input devices |
| `Apps/Console/Transcribe` | A WAV file in, the transcript and what it cost out |
| `Apps/Demo/LiveTranscribe` | A window: pick an input, watch the meter, read the transcript as it arrives |
| `Samples` | `jfk.wav`, the recording the end-to-end tests run on |
| `Tests` | NanoTest, one executable per module; `Tests/Oracle` compares against whisper.cpp |
| `Benchmark` | Our runtime against whisper.cpp, in one process, behind `WHISPER_EACP_ENABLE_BENCHMARK` |

Headers are spelled `<WhisperEACP/...>`, beside eacp's own `<eacp/...>`, and
everything lives in namespace `WSP`.

Three kinds of matrix product share the tree. `TiledMatMul` is the
register-tiled product for the encoder's many rows, `SplitLinear` splits a
decode step's few rows across a group, and `SimdTiledMatMul` is the many-row
product again out of eacp's SIMD-group matrices. The role aliases at the foot
of `Kernels/SimdTiledMatMul.h` — `LinearProduct`, `AttentionScoresProduct` and
the rest — are what every call site names, so which kernel a product runs is
one decision in one place.

## Using the library

```cpp
#include <WhisperEACP/WhisperEACP.h>

auto whisper = WSP::Whisper {};
whisper.load("path/to/whisper-tiny.en");   // or loadBundled()
whisper.prepare();                          // compiles the kernels, uploads the weights

const auto samples = WSP::readWavFile("recording.wav");   // 16 kHz mono, <= 30 s
const auto text = whisper.transcribeText(samples);
```

`load` maps the four files of a HuggingFace Whisper repo and touches no GPU;
`prepare` is the half second of kernel compilation and upload. A run longer
than the 30 s window is an error rather than a truncation, since deciding where
to cut a recording is a chunking layer that does not exist yet.

`LiveTranscriber` sits above that for audio that keeps arriving: push 16 kHz
mono samples of any block size, call `update()` from the thread that owns the
GPU, and read `committed()` for the closed segments and `pending()` for the one
still being re-decoded. Every clock in it is audio time rather than wall time,
so a run over a given recording is deterministic.

```cpp
auto live = WSP::LiveTranscriber {whisper};

live.push(block);
if (live.update())
    show(live.committed(), live.pending());
```

Everything GPU-side is main-thread only, which is eacp's rule rather than this
project's. Anything touching the device runs inside `eacp::Apps::run`, which
owns the run loop and autorelease pool the Metal backend is written against.

## Building

Requires CMake 3.31+, a C++20 compiler and a 64-bit toolchain. eacp, MakeASound
and NanoTest are fetched through CPM — eacp at `develop`, the others at `main`
— so the configure line is the whole story and no build here points at a
checkout on the machine.

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DWHISPER_EACP_UNITY_BUILD=OFF
cmake --build build

ctest --test-dir build --output-on-failure
./build/Apps/Console/DeviceInfo/DeviceInfo
```

macOS and Windows. eacp gates its whole GPU stack behind platforms with a Metal
or D3D12 backend, so there is no Linux target.

| option | default | |
| --- | --- | --- |
| `WHISPER_EACP_FETCH_MODEL` | `ON` | Download the four `tiny.en` files (151 MB) and copy them beside every binary that asks. Off, `Whisper::hasBundledModel()` answers `false` and the tests that need a file skip |
| `WHISPER_EACP_ENABLE_TESTS` | on when top-level | The `Tests/` tree |
| `WHISPER_EACP_ENABLE_APPS` | on when top-level | The `Apps/` tree. The demo needs eacp's widget tier, which eacp builds only where it has a Metal or D3D12 backend |
| `WHISPER_EACP_ENABLE_WHISPER_CPP` | `OFF` | Fetch whisper.cpp with every backend off and build `Tests/Oracle` against it |
| `WHISPER_EACP_ENABLE_BENCHMARK` | `OFF` | Fetch whisper.cpp at its own defaults for the machine and build `Benchmark` |
| `WHISPER_EACP_UNITY_BUILD` | `OFF` | Unity builds of the libraries. Off, per-file compile commands land in `compile_commands.json` |
| `WHISPER_EACP_CI_BUILD` | `OFF` | The unity builds CI uses, here and in eacp, MakeASound and Miro |

### Building against a local eacp

For the case that is actually for — changing eacp itself alongside a change
here that needs it:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DWHISPER_EACP_UNITY_BUILD=OFF \
      -DCPM_eacp_SOURCE=$HOME/Code/eacp
```

Use `$HOME`, not `~`: CMake does not expand a tilde and the shell will not
expand one inside quotes, so the path silently resolves to nothing and the
failure surfaces much later as a missing `eacp-gpu` target. A local tree also
brings whatever is uncommitted in it into the build, so an unrelated refactor in
progress there breaks every target here.

## Transcribing

The build copies the `tiny.en` weights beside `Transcribe` and embeds the
sample in it, so the first form needs nothing but the build tree:

```bash
./build/Apps/Console/Transcribe/Transcribe
./build/Apps/Console/Transcribe/Transcribe recording.wav
./build/Apps/Console/Transcribe/Transcribe path/to/whisper-tiny.en recording.wav
```

```
model: /path/to/build/Apps/Console/Transcribe/WhisperModel
audio: built-in jfk.wav

 And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.

  load and upload          1.451 s
  mel + encoder            0.020 s
  decode,  24 tokens       0.016 s     0.7 ms per step over 25
  mel to transcript        0.036 s
```

That is a Debug tree; the load is mostly kernel compilation, paid once.

The model is what makes the default configure download 151 MB. It is copied
beside the binary after every link rather than compiled into it — into
`Contents/Resources` for a macOS bundle, next to the executable otherwise — so
no translation unit ever holds it. Configure with
`-DWHISPER_EACP_FETCH_MODEL=OFF` to skip the download; the two-argument form
above still works, and `Whisper::hasBundledModel()` answers `false`.

The weights are HuggingFace **safetensors**, read straight from a repo's four
files: `model.safetensors`, `config.json`, `tokenizer.json` and
`preprocessor_config.json`, the last of which carries the mel filterbank
already built. `tiny.en` ships F32, but every value in it round-trips through
fp16 unchanged — OpenAI's checkpoints are fp16 and the conversion only widens
them — so the logits projection keeps an fp16 copy of the vocabulary and comes
out bit identical for half the bandwidth.

`Samples/jfk.wav` is 11 seconds of President Kennedy's inaugural address,
20 January 1961, at 16 kHz mono PCM16 — byte-identical to whisper.cpp v1.9.3's
`samples/jfk.wav`, which is OpenAI whisper's own `tests/jfk.flac`. The recording
is a US government work held by the JFK Library and is
[public domain](https://archive.org/details/JohnF.KennedyInauguralAddress).

### Live Transcribe

`Apps/Demo/LiveTranscribe` is the same runtime over a microphone: pick an input
device and which of its channels to listen to, watch the level meter, and read
the transcript as it is spoken — closed sentences in white, the one still being
re-decoded in blue. It carries the model the way `Transcribe` does, so it needs
nothing on disk either.

```bash
open ./build/Apps/Demo/LiveTranscribe/LiveTranscribe.app
```

Capture, the segment policy and the model all run on the message thread, since
eacp's GPU layer is main-thread only — so a run of the model is a few tens of
milliseconds the window waits for, and only the device callback is on a thread
of its own. `--autostart` opens the default input as soon as the model is ready
and logs a line a second to stdout, which is how a run is checked without a hand
on the mouse.

## Testing

NanoTest, one executable per module, and three tiers in the order a failure is
diagnosed:

1. **A scalar CPU reference inside the test itself**, for every kernel and
   every layer — layernorm, GELU, softmax, the products, attention, the
   encoder and decoder against double precision. No dependency, and the only
   tier that catches a backend divergence, since the same assertion runs
   against MSL on Apple and HLSL on Windows. This is the bulk of the suite.
2. **whisper.cpp as an in-process oracle**, `Tests/Oracle`, behind
   `WHISPER_EACP_ENABLE_WHISPER_CPP`: the mel, the logits, the tokenizer and
   the transcript, stage by stage, so a mismatch bisects to a layer rather
   than only reporting a wrong transcript.
3. **Small committed fixtures** for what is cheap and stable — the filterbank
   matrix, a handful of tokenizer cases.

A test that needs the GPU returns early when no device is valid, and one that
needs a model file skips when the fetch is off, so the suite passes on any
machine and runs in full on one with both.

## Benchmarking

`Benchmark/` times the runtime against whisper.cpp v1.9.3 in one process — on
its Metal backend and on the CPU — over the same samples, the same greedy
policy, one warm-up and the median of the timed runs. It sits behind
`WHISPER_EACP_ENABLE_BENCHMARK`, and since a benchmark of a Debug build
measures the Debug build, the numbers come from a Release tree of their own:

```bash
cmake -G Ninja -B build-release -DCMAKE_BUILD_TYPE=Release \
      -DWHISPER_EACP_ENABLE_BENCHMARK=ON
cmake --build build-release --target Benchmark
./build-release/Benchmark/Benchmark            # jfk.wav, 10 timed runs
./build-release/Benchmark/Benchmark 30 recording.wav
```

The option builds whisper.cpp at its own defaults for the machine, backends
on, where the oracle in `Tests/Oracle` alone builds it with every backend off
to be a reference; whisper.cpp's targets are global, so a tree holds one build
and the benchmark's wins when both are on. `Benchmark/README.md` says what
that means for the oracle, what each row of the table measures and, as
importantly, what it does not. `Benchmark/backends.py` runs mlx-whisper,
faster-whisper and openai-whisper — whichever are installed — on the same
protocol, for the backends that cannot be in the process.

## License

MIT — see `LICENSE`.
