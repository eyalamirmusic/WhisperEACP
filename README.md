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
came in which order, and the gaps this surfaced in eacp, Miro and ResEmbed —
the second half of the point.

## Layout

| | |
| --- | --- |
| `Lib/WhisperEACP/Core` | Shared types and the library version. Links `eacp-gpu` |
| `Lib/WhisperEACP/Audio` | Whisper's audio contract, and capture through MakeASound |
| `Lib/WhisperEACP/Kernels` | layernorm, GELU, softmax, matmul — shapes as uniforms |
| `Lib/WhisperEACP/Mel` | Hann, STFT, the 80 x 201 filterbank, log10 and the scaling |
| `Lib/WhisperEACP/Model` | safetensors weights, `config.json`, the filterbank |
| `Lib/WhisperEACP/Tokenizer` | byte-level BPE and Whisper's special tokens |
| `Lib/WhisperEACP/Encoder` | HF's `WhisperEncoder.forward`, recorded into one command buffer |
| `Lib/WhisperEACP/Decoder` | The same for the text half, over a KV cache |
| `Lib/WhisperEACP/Whisper` | The whole runtime — samples in, a transcript out |
| `Apps/Console/DeviceInfo` | What this machine offers: GPU limits and input devices |
| `Apps/Console/Transcribe` | A WAV file in, the transcript and what it cost out |
| `Samples` | `jfk.wav`, the recording the end-to-end tests run on |
| `Tests` | NanoTest, one executable per module |

Headers are spelled `<WhisperEACP/...>`, beside eacp's own `<eacp/...>`.

## Building

Requires CMake 3.31+, a C++20 compiler and a 64-bit toolchain. eacp, MakeASound
and NanoTest are fetched automatically via CPM.

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DWHISPER_EACP_UNITY_BUILD=OFF
cmake --build build

ctest --test-dir build --output-on-failure
./build/Apps/Console/DeviceInfo/DeviceInfo
```

macOS and Windows. eacp gates its whole GPU stack behind platforms with a Metal
or D3D12 backend, so there is no Linux target.

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
```

The model is what makes the default configure download 151 MB. It is copied
beside the binary after every link rather than compiled into it — into
`Contents/Resources` for a macOS bundle, next to the executable otherwise — so
no translation unit ever holds it. Configure with
`-DWHISPER_EACP_FETCH_MODEL=OFF` to skip the download; the two-argument form
above still works, and `Whisper::hasBundledModel()` answers `false`.

`Samples/jfk.wav` is 11 seconds of President Kennedy's inaugural address,
20 January 1961, at 16 kHz mono PCM16 — byte-identical to whisper.cpp v1.9.3's
`samples/jfk.wav`, which is OpenAI whisper's own `tests/jfk.flac`. The recording
is a US government work held by the JFK Library and is
[public domain](https://archive.org/details/JohnF.KennedyInauguralAddress).

eacp is fetched at `develop`, MakeASound and NanoTest at `main`. The configure
line above is the whole story — no build here points at a checkout on the
machine.

To build against a local eacp checkout instead, for the case that is actually
for — changing eacp itself alongside a change here that needs it:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DWHISPER_EACP_UNITY_BUILD=OFF \
      -DCPM_eacp_SOURCE=$HOME/Code/eacp
```

Use `$HOME`, not `~`: CMake does not expand a tilde and the shell will not
expand one inside quotes, so the path silently resolves to nothing and the
failure surfaces much later as a missing `eacp-gpu` target. A local tree also
brings whatever is uncommitted in it into the build, so an unrelated refactor in
progress there breaks every target here.
