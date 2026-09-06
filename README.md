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

Early days — the repository is set up, the toolchain is proven end to end by
`Tests/GPU`, and the model layers are not written yet.

## Layout

| | |
| --- | --- |
| `Lib/WhisperEACP/Core` | Shared types and the library version. Links `eacp-gpu` |
| `Lib/WhisperEACP/Audio` | Whisper's audio contract, and capture through MakeASound |
| `Apps/Console/DeviceInfo` | What this machine offers: GPU limits and input devices |
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

To build against a local eacp checkout — the normal setup here, since this
project drives changes to eacp's compute layer:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DWHISPER_EACP_UNITY_BUILD=OFF \
      -DCPM_eacp_SOURCE=$HOME/Code/eacp -DCPM_MakeASound_SOURCE=$HOME/Code/makeasound
```

Use `$HOME`, not `~`: CMake does not expand a tilde and the shell will not
expand one inside quotes, so the path silently resolves to nothing and the
failure surfaces much later as a missing `eacp-gpu` target.
