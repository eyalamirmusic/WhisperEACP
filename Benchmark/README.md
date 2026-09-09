# Benchmark

Our runtime against whisper.cpp, in one process, on the same samples. Part of
the main tree behind `WHISPER_EACP_ENABLE_BENCHMARK`, which also decides
which whisper.cpp the tree builds — see below.

## Running it

A benchmark of a Debug build measures the Debug build, and the tree's usual
configure is Debug, so the numbers come from a Release tree of their own:

```bash
cmake -G Ninja -B build-release -DCMAKE_BUILD_TYPE=Release \
      -DWHISPER_EACP_ENABLE_BENCHMARK=ON
cmake --build build-release --target Benchmark
./build-release/Benchmark/Benchmark
```

The option fetches whisper.cpp v1.9.3 and its `ggml-tiny.en.bin` (75 MB); the
HuggingFace `tiny.en` is copied beside the binary as it is beside
`Transcribe`. The binary prints the configuration it was built in above its
numbers, so a run out of the Debug tree says so rather than passing for a
measurement.

```
Benchmark                  Samples/jfk.wav, 10 timed runs
Benchmark 30               30 timed runs
Benchmark recording.wav    another 16 kHz mono WAV of at most 30 s
Benchmark 30 recording.wav
Benchmark 30 --audio-ctx=704       both sides over 704 of the 1500 encoder positions
Benchmark 30 --audio-ctx=audio     the positions Whisper::audioContextForSamples picks

Benchmark --live           jfk.wav on repeat, 30 s of stream, a 1.5 s gap
Benchmark --live 60        60 s of stream
Benchmark --live 60 4      a 4 s gap between passes; 0 is speech with no pause in it
Benchmark --live 60 4 recording.wav
Benchmark --live --audio-ctx=audio    the live loop encoding only the audio it holds
```

`--live` is the other measurement, and it has a section of its own below.

## Which whisper.cpp

whisper.cpp's targets are named globally, so a tree holds one build of it,
and two things here want different ones. `Tests/Oracle` wants every backend
off, so that the reference computes exactly ggml's own CPU path. This wants
whisper.cpp's own defaults for the machine — Metal, Accelerate and BLAS on a
Mac — because a benchmark against a handicapped whisper.cpp measures nothing.

The root `CMakeLists.txt` makes the call: the benchmark wins whenever it is
on. The oracle tests in the same tree then run against that build with its
GPU off at runtime, which is ggml's CPU backend plus Accelerate through the
BLAS one; their tolerances are measured against the reference's own
resolution rather than assumed, so they hold either way, and they print which
build they loaded. A tree with only `WHISPER_EACP_ENABLE_WHISPER_CPP` on keeps
the every-backend-off oracle.

## What it measures

Three contestants, each loaded once, each given one warm-up run and then the
timed runs, of which the **median** is what the columns compare:

| column | what runs |
| --- | --- |
| `WhisperEACP` | our runtime, on eacp's device, from the F32 safetensors |
| `whisper.cpp GPU` | whisper.cpp on its GPU backend — Metal on a Mac — from the F16 GGML file |
| `whisper.cpp CPU` | the same context with `use_gpu` off: ggml's CPU backend, plus Accelerate through the BLAS backend on a Mac, at whisper.cpp's own default thread count |

The header names whisper.cpp's version, its flash-attention default and every
backend ggml registered, so a number is never read without knowing what
produced it. On a build with no GPU backend the middle column is skipped and
says so.

The decode is the same policy on every side — greedy, temperature 0, no
fallback, one segment, no timestamps, HF's non-speech suppression — spelled
for whisper.cpp exactly as `Tests/Oracle` spells it, so the transcripts are
comparable token for token. The `transcript` row says whether each
whisper.cpp column decoded the same tokens ours did.

The rows:

| row | |
| --- | --- |
| `transcribe, median` / `best` | wall clock around the whole call, samples in to tokens out. The one number that is defined the same way on every side |
| `x real time, 30 s window` | the window's length over the median. Every encoder here does the work of a full window whatever the clip's length, so the window is the honest denominator; an 11 s clip does not transcribe three times faster than a 30 s one |
| `encode, median` | ours: the upload, the mel and the encoder, which are one command buffer. whisper.cpp: its encoder alone, since `whisper_get_timings` does not expose its mel. Read this row knowing it |
| `decode, median` | the prompt pass and every token after it, on both sides. Ours is the pipelined loop end to end — the first step's submit to the read of the last token — and stops at that read rather than waiting for the one step still in the air behind it |
| `decode per step, median` | that over `steps`, which counts one for the prompt and one per sampled token on both sides. A step is counted when its token is read, so the step a run leaves running past its end is in neither number |

whisper.cpp reports its stages as per-call averages without the call counts,
so its decode total is put back from the shape of a temperature-0 greedy run:
one prompt pass, then one single-token decode per text token. That is exact
for the policy above.

Every clock is wall clock from the host. eacp has no GPU timestamp hook (the
gaps table in `plan.md`), and whisper.cpp's are `ggml_time_us` around the
same boundaries.

## The live mode

`--live` measures the other thing this runtime is asked to do: not one
transcription, but keeping up with a microphone. `LiveTranscriber` re-runs the
open segment every `stepSeconds` of new audio, so a machine that is listening
pays a full transcribe several times a second, and what it costs is a *share of
the wall clock* rather than a duration.

The recording is played on repeat with a gap of silence between passes — a
microphone left open over somebody who says the same thing again after a pause
— pushed at a 33 ms tick, which is the timer `Apps/Demo/LiveTranscribe` runs.
A run that overruns its tick hands the next one a larger block, exactly as the
capture queue does while the model has the thread. No whisper.cpp on the other
side: whisper.cpp has no equivalent of this layer, so there is nothing to put
in a second column.

Every clock in `LiveTranscriber` is audio time, so `runs` is the same number
every time for a given recording, gap and policy, whatever the machine was
doing. That is what makes this a before-and-after tool: the run count moves
only when the policy does, and the seconds under it are the noise.

```
  stream, wall clock                       30.0 s
  runs                                         49
  runs that changed the text                   39
  model                                    1.280 s
  duty                                        4.3 %
  per run, mean                             26.1 ms
  per run, longest                          35.3 ms
  encode, mean                              14.2 ms
  decode, mean                              11.0 ms
  steps per run                              13.0
  segments committed                            2
```

`--audio-ctx=audio` runs the stream with `LiveOptions::encodeOnlyTheAudioThereIs`
on, which is what `Apps/Demo/LiveTranscribe` does: each run encodes the
segment's audio plus a margin rather than the whole window, and the policy line
above the table says which of the two a run measured. In the comparison mode the
same flag sizes the context to the recording, on both sides.

| row | |
| --- | --- |
| `runs` | `Whisper::transcribe` calls the policy asked for. This is the number the policy moves; everything under it is what one of them costs |
| `runs that changed the text` | of those, the ones whose transcript differed from the run before — a run over audio that told the model nothing is a run that should not have happened |
| `model` | the wall clock inside those calls, added up |
| `duty` | that over the stream's wall clock: **the number this mode exists for**, model seconds per wall second |
| `per run` | the mean and the longest of the same calls. The longest is what a tick can be blocked for, which is the UI's latency rather than the machine's load |
| `encode` / `decode` / `steps per run` | the same stages the table above breaks out, averaged over the runs |
| `segments committed` | lines `LiveTranscriber` closed and handed to `committed()`, and the first of them is printed under the table so a policy change that broke the transcript says so |

Two things have to be read with it, and both are ways of being wrong about
this number.

**A run at live cadence costs about twice what the table above measures.** The
comparison runs back to back and holds the GPU at its clock; half a second of
idle between 26 ms bursts does not, so the same work that takes ~21 ms there
takes ~26 ms here, and the first run after a longer gap more. That is a real
cost of running live, not a measurement artefact, which is why this mode times
the runs it actually took rather than reusing the median from above.

**macOS's GPU utilisation counter cannot see this.** The `AGXAccelerator`
`Device Utilization %` in `ioreg`, which is what Activity Monitor shows, is a
short-window snapshot: a 26 ms burst every 500 ms reads there as 50-80% busy
while the true share is a twentieth. The duty row is the honest form of the
same question, and only the runtime can answer it, since only the runtime knows
which wall clock was its own.

## What it does not measure

- **Other Whisper runtimes.** mlx-whisper, faster-whisper and openai-whisper
  are Python and cannot be in this process. `backends.py` runs whichever of
  them is installed on the same protocol — same samples, same policy, one
  warm-up, median of the timed runs — and prints a table whose one row,
  transcribe wall clock, is the top row of this one:

  ```bash
  pip install mlx-whisper faster-whisper openai-whisper   # any subset
  python3 Benchmark/backends.py --runs 10
  ```

  None of the three is a dependency of this project, and the script skips a
  missing one with a line.

- **Anything but `tiny.en`** and, by default, anything but `jfk.wav`. The
  model is the one the build fetches; another WAV is an argument.

- **The microphone**, in `--live`. The samples come from a file at a timer's
  pace, so `Audio/Capture`, its device callback, its queue and its resampling
  are all outside the measurement — which is deliberate: the numbers are then
  the same on any machine with any input device, and reproducible on one with
  none. `LiveTranscriber` is what the demo app puts between the two, and it is
  what this drives.

- **Whether the live transcript is any good**, beyond printing the first line
  it committed. A policy that runs less often commits the same text or it does
  not, and `Tests/Whisper/LiveTranscriberTests.cpp` is where that is asserted
  rather than eyeballed.

- **Kernel-level cost.** The step's GPU time is a chain of some sixty small
  kernels run one after another; `plan.md`'s performance rounds say which
  ones and what each costs. Per-kernel numbers come from eacp's labelled
  passes, 128 to a command buffer, not from this table.
