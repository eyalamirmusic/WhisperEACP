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

The option fetches whisper.cpp v1.9.3 and its `ggml-tiny.en.bin` (75 MB) at
configure time; the HuggingFace `tiny.en` this side loads is fetched at startup
as it is by `Transcribe`, into a directory the two share, and the header above
the numbers says which file each side read. The binary prints the configuration it was built in above its
numbers, so a run out of the Debug tree says so rather than passing for a
measurement.

```
Benchmark                  Samples/jfk.wav, 10 timed runs
Benchmark 30               30 timed runs
Benchmark recording.wav    another 16 kHz mono WAV of at most 30 s
Benchmark 30 recording.wav
Benchmark 30 --audio-ctx=704       both sides over 704 of the 1500 encoder positions
Benchmark 30 --audio-ctx=audio     the positions Whisper::audioContextForSamples picks
Benchmark 30 --units=all           the ANE column's encoder under Core ML's all (the GPU)
Benchmark 30 --plan                also print where Core ML placed the encoder (~14 s)

Benchmark --live           jfk.wav on repeat, 30 s of stream, a 1.5 s gap
Benchmark --live 60        60 s of stream
Benchmark --live 60 4      a 4 s gap between passes; 0 is speech with no pause in it
Benchmark --live 60 4 recording.wav
Benchmark --live --step=2          re-transcribe the open segment every 2 s of audio
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

Four contestants, each loaded once, each given one warm-up run and then the
timed runs, of which the **median** is what the columns compare:

| column | what runs |
| --- | --- |
| `WhisperEACP` | our runtime, on eacp's device, from the F32 safetensors |
| `WhisperEACP ANE` | the same with the encoder on Core ML (`Whisper::setEncoderBackend(coreML)`), compiled for `--units`, `cpuAndNeuralEngine` by default; the decoder stays on eacp's device. Only where the build and the OS have Core ML |
| `whisper.cpp GPU` | whisper.cpp on its GPU backend — Metal on a Mac — from the F16 GGML file |
| `whisper.cpp CPU` | the same context with `use_gpu` off: ggml's CPU backend, plus Accelerate through the BLAS backend on a Mac, at whisper.cpp's own default thread count |

The header names whisper.cpp's version, its flash-attention default and every
backend ggml registered, so a number is never read without knowing what
produced it; and for the ANE column the compute units, whether its load found
the compiled model in the cache, and the load time. `--plan` adds where Core ML
placed the encoder's ops, which is behind a flag because reading the plan
under the engine settings is the engine compile again, about 14 s. On a build with no GPU backend the middle column is skipped and
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
| `mel on the GPU` / `predict` / `seam copies` | the ANE column's encode split three ways: the mel's own command buffer, committed and waited on, which is all the GPU does in that encode; the Core ML prediction alone; and the rest, per run, which is the mel read back and narrowed to fp16 and the rows widened back into the decoder's buffer. `-` in the other columns |
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
the time it is listening* rather than a duration.

The recording is played on repeat with a gap of silence between passes — a
microphone left open over somebody who says the same thing again after a pause
— pushed at a 33 ms tick, which is the timer `Apps/Demo/LiveTranscribe` runs.

**Every contestant takes that stream**, one after another, each loaded once,
and the table has the columns the comparison has. whisper.cpp has no
equivalent of this layer, so what its two columns measure is *our* policy
driving `whisper_full`: `LiveTranscriber` transcribes through a function of the
open segment rather than through a `Whisper` — a
`std::function<std::string(Span<const float>)>`, or one answering an
`eacp::Threads::Async<std::string>` — and the constructor that takes the
runtime is the second over `Whisper::transcribeAsync`. So the segments, the
runs and the audio each run saw are the policy's on every side, and only the
seconds are the backend's.

Ours streams twice, once with the encoder on the kernels and once on Core ML
(the `WhisperEACP ANE` column, where the machine has it). On Core ML a run comes
back on a later turn of the event loop, which the stream pumps between ticks,
and the transcriber takes no audio until it does, so its `runs` can fall short
of the others' when a run outlasts the step; and that column's `per run`
and `duty` are each run's start-to-result latency, loop turns included, rather
than time the main thread or the GPU spent on it. With
`--audio-ctx=audio` the per-run context is set on each side from the same
`Whisper::audioContextForSamples(segment)` — `Whisper::setAudioContext` for
ours, `whisper_full_params::audio_ctx` for theirs.

Every clock in `LiveTranscriber` is audio time, and the tick delivers a tick's
worth of samples rather than however many the wall clock ran past, so `runs` is
the same number for a given recording, gap and policy whatever ran the model.
That is what makes the columns comparable, and what makes this a
before-and-after tool: the run count moves only when the policy does, and the
seconds under it are the noise. A contestant that cannot keep up falls behind
the clock rather than skipping runs, and the two stream rows are where that
shows — `stream, wall clock` longer than `stream, audio` is a backend that took
longer than the audio it was listening to.

```
                                      WhisperEACP    whisper.cpp GPU    whisper.cpp CPU
  stream, audio                            30.0 s             30.0 s             30.0 s
  stream, wall clock                       30.0 s             30.0 s             30.0 s
  runs                                         49                 49                 49
  runs that changed the text                   39                 39                 39
  model                                   1.234 s            2.910 s            6.155 s
  duty                                      4.1 %              9.7 %             20.5 %
  per run, mean                           25.2 ms            59.4 ms           125.6 ms
  per run, longest                        50.5 ms           107.0 ms           139.9 ms
  encode, mean                            13.9 ms            15.1 ms           102.4 ms
  decode, mean                            10.3 ms            18.3 ms             7.8 ms
  steps per run                              13.0               13.0               13.0
  segments committed                            2                  2                  2
  transcript                            reference               same               same
```

`--audio-ctx=audio` runs the stream with `LiveOptions::encodeOnlyTheAudioThereIs`
on, which is what `Apps/Demo/LiveTranscribe` does: each run encodes the
segment's audio plus a margin rather than the whole window, and the policy line
above the table says which of the two a run measured. In the comparison mode the
same flag sizes the context to the recording, on both sides. `--step=<seconds>`
is `LiveOptions::stepSeconds`, how much new audio the open segment takes before
it is transcribed again, and it is the one policy knob this mode exposes because
it is the one that decides how many runs a minute of speech costs.

| row | |
| --- | --- |
| `stream, audio` / `wall clock` | the audio the policy was handed, and how long that took. Equal while a contestant keeps up |
| `runs` | `transcribe` calls the policy asked for — the same number in every column. This is the number the policy moves; everything under it is what one of them costs |
| `runs that changed the text` | of those, the ones whose transcript differed from the run before — a run over audio that told the model nothing is a run that should not have happened. It may differ between columns, since it is about what each backend decoded |
| `model` | the wall clock inside those calls, added up |
| `duty` | that over the stream's audio: **the number this mode exists for**, model seconds per second listened to. Over 100% is a backend that cannot keep up at this policy |
| `per run` | the mean and the longest of the same calls. The longest is what a tick can be blocked for, which is the UI's latency rather than the machine's load |
| `encode` / `decode` / `steps per run` | the same stages the table above breaks out, averaged over the runs, and read with the same caveat — ours includes the mel, whisper.cpp's does not |
| `segments committed` | lines `LiveTranscriber` closed and handed to `committed()`, and the first of each column's is printed above the table so a policy change that broke the transcript says so |
| `transcript` | whether a column committed the same lines ours did, as in the comparison table |

Two things have to be read with it, and both are ways of being wrong about
this number.

**A run at live cadence costs more than the comparison above measures, and how
much more is what `--step` decides.** The comparison runs back to back and holds
the GPU at its clock; seconds of idle between 25 ms bursts do not. jfk.wav on
repeat, a 1.5 s gap, 30 s of stream, the whole window on every side:

| `--step` | 0.5 s | 1.0 s | 1.5 s | 2.0 s |
| --- | --- | --- | --- | --- |
| runs | 49 | 31 | 21 | 17 |
| `WhisperEACP`, per run | 25.2 ms | 23.8 ms | 42.2 ms | 49.8 ms |
| `whisper.cpp GPU`, per run | 59.4 ms | 50.3 ms | 58.5 ms | 61.9 ms |
| `whisper.cpp CPU`, per run | 125.6 ms | 121.9 ms | 124.1 ms | 123.8 ms |

The same transcribe costs us 1.8x more with the runs 1.5 s apart than 1 s
apart, and 2.8x what the comparison mode's 18 ms measures — our encode and our
decode both roughly double, which is a GPU that clocked down between bursts
rather than anything the policy did. **whisper.cpp's Metal column barely shows
it**: its encode does the same thing (14.4 ms to 22.6 ms across the same steps)
but its decode is flat and its decode is most of its run, so its total moves
1.2x. The CPU column does not move at all, which is the control. So a wider step
buys our runtime nothing past a second — 31 runs at 23.8 ms is less model time
than 21 at 42.2 ms — and that is a real cost of running live rather than a
measurement artefact, which is why this mode times the runs it actually took
instead of reusing the median from above.

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
  model is the one each side fetches — ours at startup, theirs at configure
  time; another WAV is an argument.

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
