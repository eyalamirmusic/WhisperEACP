#!/usr/bin/env python3
"""The Python Whisper backends, on the protocol Benchmark runs in C++.

Benchmark times our runtime against whisper.cpp in one process, and these
three cannot be in it: mlx-whisper, faster-whisper and openai-whisper are
Python packages with their own runtimes. This runs whichever of them is
installed over the same input, the same way — tiny.en, greedy at temperature 0,
no timestamps, one warm-up and a set of timed runs of which the median is
reported — and prints a table of the same shape, so a row of it can sit next to
a column of Benchmark's.

    pip install mlx-whisper          # Apple silicon, Metal through MLX
    pip install faster-whisper       # CTranslate2, CPU on a Mac
    pip install openai-whisper       # PyTorch, the original

    python3 Benchmark/backends.py [--runs N] [--wav FILE] [--device DEV]

None of the three is a dependency of this project, and the script imports each
one lazily and skips it with a line when it is missing. numpy is what every one
of them needs anyway, and is the one import at the top.

What is comparable and what is not: every backend here is timed by wall clock
around its whole transcribe call, which is the "transcribe, median" row of
Benchmark's table and the only row this one has. The stage numbers Benchmark
prints have no counterpart these libraries expose the same way.
"""

import argparse
import os
import statistics
import sys
import time
import wave

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SECONDS = 30
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WAV = os.path.join(HERE, "..", "Samples", "jfk.wav")

DEFAULT_RUNS = 10
WARM_UP_RUNS = 1


def read_window(path):
    """16 kHz mono PCM16, as float32, zero-filled to the 30 s window — the
    same samples Benchmark hands both of its sides, for the same reason: the
    window is what every encoder here does the work of, and an unpadded input
    reaches different front-ends as different mels."""
    with wave.open(path, "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE:
            raise SystemExit(f"{path} is {wav.getframerate()} Hz, not {SAMPLE_RATE}")
        if wav.getsampwidth() != 2:
            raise SystemExit(f"{path} is not 16-bit PCM")
        frames = wav.readframes(wav.getnframes())
        channels = wav.getnchannels()

    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0]

    if len(pcm) > WINDOW_SAMPLES:
        raise SystemExit(f"{path} is longer than the {WINDOW_SECONDS} s window")

    samples = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    samples[: len(pcm)] = pcm.astype(np.float32) / 32768.0
    return samples, len(pcm) / SAMPLE_RATE


class Backend:
    """One library: a name for the table, a load, and a transcribe that
    returns the text. load() is where the import happens, so a missing
    package is reported by name rather than as a traceback."""

    name = ""

    def load(self, device):
        raise NotImplementedError

    def transcribe(self, samples):
        raise NotImplementedError


class MlxWhisper(Backend):
    name = "mlx-whisper"
    model = "mlx-community/whisper-tiny.en-mlx"

    def load(self, device):
        import mlx_whisper

        self.module = mlx_whisper
        self.name = f"mlx-whisper ({self.model})"

    def transcribe(self, samples):
        return self.module.transcribe(
            samples,
            path_or_hf_repo=self.model,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
        )["text"]


class FasterWhisper(Backend):
    name = "faster-whisper"
    model = "tiny.en"

    def load(self, device):
        from faster_whisper import WhisperModel

        compute = "float32"
        self.model_object = WhisperModel(
            self.model, device=device or "cpu", compute_type=compute
        )
        self.name = f"faster-whisper ({self.model}, {device or 'cpu'}, {compute})"

    def transcribe(self, samples):
        segments, _ = self.model_object.transcribe(
            samples,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments)


class OpenAiWhisper(Backend):
    name = "openai-whisper"
    model = "tiny.en"

    def load(self, device):
        import whisper

        self.model_object = whisper.load_model(self.model, device=device or "cpu")
        self.name = f"openai-whisper ({self.model}, {device or 'cpu'}, fp32)"

    def transcribe(self, samples):
        return self.model_object.transcribe(
            samples,
            temperature=0.0,
            beam_size=None,
            best_of=None,
            without_timestamps=True,
            condition_on_previous_text=False,
            fp16=False,
        )["text"]


BACKENDS = [MlxWhisper, FasterWhisper, OpenAiWhisper]


def benchmark(backend, samples, runs):
    text = ""
    for _ in range(WARM_UP_RUNS):
        text = backend.transcribe(samples)

    wall = []
    for _ in range(runs):
        start = time.perf_counter()
        backend.transcribe(samples)
        wall.append(time.perf_counter() - start)

    return statistics.median(wall), min(wall), text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--wav", default=DEFAULT_WAV)
    parser.add_argument(
        "--device",
        default=None,
        help="faster-whisper's and openai-whisper's device; cpu by default",
    )
    arguments = parser.parse_args()

    samples, audio_seconds = read_window(arguments.wav)

    print("WhisperEACP benchmark, Python backends")
    print(
        f"  audio {arguments.wav}: {audio_seconds:.1f} s, "
        f"zero-filled to the {WINDOW_SECONDS} s window"
    )
    print(f"  runs  {arguments.runs} timed after {WARM_UP_RUNS} warm-up, per backend")
    print()

    results = []
    for backend_type in BACKENDS:
        backend = backend_type()
        try:
            backend.load(arguments.device)
        except ImportError as missing:
            print(f"  {backend.name}: not installed ({missing.name})")
            print()
            continue

        median, best, text = benchmark(backend, samples, arguments.runs)
        results.append((backend.name, median, best, text))
        print(f"  {backend.name}")
        print(f"   {text}")
        print()

    if not results:
        print("  none of the three backends is installed; see the top of this file")
        return 2

    width = max(len(name) for name, *_ in results) + 2
    print(f"  {'':{width}}{'transcribe, median':>20}{'best':>12}{'x real time':>14}")
    for name, median, best, _ in results:
        print(
            f"  {name:{width}}{median:>18.3f} s{best:>10.3f} s"
            f"{WINDOW_SECONDS / median:>14.1f}"
        )
    print()
    print(f"  x real time is over the {WINDOW_SECONDS} s window, as in Benchmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
