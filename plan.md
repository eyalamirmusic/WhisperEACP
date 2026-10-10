# Plan

What is left to do in WhisperEACP, and the order it is worth doing in. The
record of how the tree got here, every performance round and the Core ML
work through "Everything on the Neural Engine", is the previous `plan.md` in
git history (last at `4dd7e5b`); eacp's side of the Core ML work is the
"Design record" in its `Lib/eacp/ML/README.md`. The rule the old plan kept
still holds: a gap belongs in eacp, not in a workaround here.

## Where the ground truth lives

| question | answer |
| --- | --- |
| audio contract | `Lib/WhisperEACP/Audio/Format.h`, pinned by value in `Tests/Audio` |
| model shapes | `config.json` from the HF repo |
| mel filterbank | `preprocessor_config.json`, an 80 x 201 matrix |
| weights | `model.safetensors`, fetched at run time |
| vocabulary | `tokenizer.json` |
| the transcript | the oracle in `Tests/Oracle`, whisper.cpp's tokens on jfk.wav |

## Where it stands

The kernels are the default and the fastest: 14.4 ms a transcribe of jfk.wav
on an M5 Max, the encode 5.0 ms and a step 360 µs. Core ML is a backend a
run chooses, per part: the encoder (`setEncoderBackend`, `cpuAndNeuralEngine`
by default, the only setting that reaches the engine), the mel in front of it
(`coreMLWithMel`), and the decoder (`setDecoderBackend`, `cpu` by default,
the fastest Core ML step by a third). Everything on the engine
(`setNeuralEngineThroughout`) is 2.6x the kernels' wall time and the one
configuration that leaves the GPU at 0% and the CPU nearly idle; the engine
encoder alone leaves the GPU free for the length of its prediction and costs
7 ms of wall time at the full window. Under `all`, Core ML's own GPU lowering
of the encoder is three times faster than our Metal kernels. The Core ML
decoder opens a sequence with its prompt in the start's one prediction, so it
makes as many predictions as the kernels make steps; that took everything on
the engine from 48.4 to 47.0 ms a transcribe of jfk.wav in Release (the
kernels 17.2 ms in the same tree), its decode from 33.8 to 32.5 ms. The
transcript is the reference's in every configuration. The compiled Core ML models live
in eacp's shared, bounded cache, `~/Library/Caches/eacp/CoreML`, one folder
for every binary of the tree since each is stamped with the company `eacp`
(the tests keep their own under the temporary directory).

Where Core ML would put the encoder's enumerated shapes on a CPU that traps
(before macOS 27, under `cpu`, or `cpuAndNeuralEngine` with no engine), the
encoder backend is refused up front: `supportsEncoderBackend(backend, units)`
is false, `encoderBackendRefusal` says why, and the setters and switches throw
the `ModelError` before anything is built, the kernels running as before.
`setCoreMLPlatform` replaces eacp's two predicates, which is how
`Tests/Whisper/RefusalTests.cpp` stands in for such a machine.

## Open, in the order it is worth doing

1. **The fused mel-and-encoder model**, one prediction from samples to rows,
   which would also remove the hop back to the main thread between the two
   predictions of the async encode from samples. It waits on eacp: a graph
   with an enumerated input and a fixed one beside it falls to the CPU
   (E5RT refuses the fixed input's strides) and Core ML allows only one
   enumerated input; `Graph::slice` refuses a range on an axis an enumerated
   input left unknown, which is what would let one enumerated input carry
   both the samples and the context; and no `gather` is placed on the
   engine. Each is an eacp change first.

2. **A Core ML decoder per audio context.** What a run on a reduced context
   pays on the Core ML decoder is mostly the step's, not the start's: the
   step attends over all 1500 cross rows behind `cross_mask` whatever the
   context. Measured in Release, a decoder of each shape built with the
   prompt in its start, median of 41 sequences of 24 steps, three runs that
   agreed within 0.1 ms:

   | cross rows | start, `cpu` | step, `cpu` | start, engine | step, engine |
   | --- | --- | --- | --- | --- |
   | 1500 | 2.06 ms | 0.75 ms | 1.53 ms | 1.24 ms |
   | 704 | 1.33 ms | 0.63 ms | 1.26 ms | 1.07 ms |
   | 352 | 0.94 ms | 0.58 ms | 1.13 ms | 1.01 ms |

   The start alone at 704 rows saves 0.7 ms on the CPU and 0.3 ms on the
   engine, 3% of the CPU's 20 ms decode and 1% of the engine's 32 ms, before
   the copy of its rows into the step's 1500-row cache arrays takes some of
   it back: not worth an enumerated start, which on `cpu` units would also
   need `enumeratedShapesRunOnTheCpu()`. Start and step together at 704 rows
   save about 3.6 ms over jfk.wav's start and 24 steps on the CPU and 4.4 ms
   on the engine. That wants both programs fixed at the context, compiled on
   first use and kept per context: 0.37 s a compile under `cpu` and 1.6 s
   under `cpuAndNeuralEngine`, measured, for the pair. Fixed shapes, so the
   encoder's CPU trap does not arise; the cost is a compile at each new
   context a live session reaches, which `setAudioContext`'s audio sizing
   would have to round to a few contexts to bound. Only for a run that chose
   the Core ML decoder and a reduced context.

3. **The kernels, against Core ML's GPU lowering.** The same program runs
   the full window in 1.5 ms through Core ML on the GPU against the kernels'
   5.0 ms encode: the attention and the projections at 1500 rows have a
   measured 3x target. From the fifth round, what is still on the table at
   a reduced context: the mel is the encode (1.9 of 3.0 ms at 704 positions,
   the framing, power, peak and normalise passes over 3000 frames); the
   cross attention at 115 µs a step, bound by the partial's transpose;
   chained dispatch overhead, ~140 µs across 61 dispatches, where the only
   lever is making dispatches independent; the fused encoder attention with
   its budget worked out in the old plan; and a run at live cadence costing
   twice a warm one, which only less work per run answers. The logits
   projection reads its weight at the machine's bandwidth and is done.

4. **F16, levels 2 and 3.** Level 1, fp16 weights, is done and measured.
   Level 2 is the cross-attention KV cache in fp16, measured alone (18.4 MB
   read a step, half of it saved, and not free: the cache is projected at
   run time, so storing it narrowed is a real approximation), then the
   scores, expected to be handed back. Level 3, half arithmetic, only if the
   benchmark after level 2 says the ALU is the limit, and it starts in eacp
   (a `Half` family in `ValueType`, `simdgroup_half8x8` on `SimdMatrix`, a
   half `Shared<>` tile).

Every row a Release tree of its own, A B B A against the build before it,
and a transcript compared token for token against both whisper.cpp columns
at every step; a faster wrong answer is not a result.

## Gaps for eacp

- The three under item 1: a fixed input beside an enumerated one, `slice`
  on an unknown axis, and `gather` on the engine.
- `MLComputePlan` is the program's and takes no shape, so the placement of
  one enumerated member can only be probed by a program fixed at that
  context (eacp's G4); the benchmark's `--plan` reads the program's.
- Level 3's half types, when the numbers ask for them.
- A process that holds a `Model` in a static aborts at exit: `PinnedModels`,
  the registry of the cache entries live models hold, is a function-local
  static first built by a load, so it is destroyed before a static `Model`
  built earlier, and that model's `~CacheUse` then locks a destroyed mutex
  ("mutex lock failed: Invalid argument"). `Tests/Net` keeps its encoders in a
  static map and every one of its Core ML tests passes and then aborts. The
  registry wants to outlive every `Model`: leaked (`*new PinnedModels`), or
  held through a `shared_ptr` each `CacheUse` keeps.
- `enumeratedShapesRunOnTheCpu()` and `isCpuOnly()` answer for this machine
  alone, so the refusal they drive is tested here by replacing them
  (`CoreMLPlatform`); eacp's own `Model::load` refusal has no such seam and
  is untested on a macOS 27 machine with an engine.
