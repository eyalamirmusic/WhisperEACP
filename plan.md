# Plan: F16

What it takes to run the model in fp16, split into the three things that word
can mean, in the order they pay. The earlier rounds' history — what came in
which order, and the gaps each surfaced in eacp, Miro and ResEmbed — left the
tree with the readme; this is the plan for what comes next.

## Where it stands

**Level 1 is done, and measured.** Every projection weight of a run is packed
fp16: `Whisper::setPacksWeights`, on by default, is one `WeightPacking` policy
handed to both weight structs, and `TensorLoader::loadProjectionWeight` goes
through `SafeTensors::makeExactHalfBuffer` under it, falling back to the float
upload where narrowing would lose a bit — which for a Whisper repo is nowhere,
and `Tests/Encoder` and `Tests/Decoder` assert against tiny.en that the
fallback never fired. Each linear program exists in a float-reading and a
half-reading form, and `Encoder::encodeLinear` and `Decoder::encodeLinear`
pick between them from `TensorBuffer::isPackedHalf`. Every activation, both KV
caches and the cross-attention keys and values are still fp32 in flight.

**An fp16 repo loads.** `TensorLoader::loadFloatTensor` uploads a widened
float copy of an F16 tensor — the layer norms, every bias, both convolutions
and both embedding tables — through `SafeTensors::makeFloatBuffer`, instead of
refusing it. The rule it enforced, that a packed buffer never reaches a float
subscript, now holds by construction rather than by error. `whisper-base.en`
converted to a genuine all-F16 file (HuggingFace ships that repo F32 too, it
turns out; 245 tensors, every one exact through fp16) loads and transcribes
jfk.wav to the same bytes as the F32 original.

**eacp has half storage and no half type.** `InputBuffer::readHalf`,
`readHalf2` and `ComputeProgram::writeHalf2` move packed halves through a
buffer; `ValueType` stops at `Float`, and `SimdMatrix` loads from a
`Shared<Float>` or a float buffer and emits `simdgroup_float8x8` only. Level 2
is written against what is there; Level 3 is where that changes, and it is
eacp work.

## Why storage, and not arithmetic

The runtime is bound by bytes, not multiplies. The rounds that got the step
from 590 us to 492 us did it by halving reads — the logits weight from 80 MB
to 40 MB was the largest single win a step has had — and the fourth round's
SIMD-group products bought the encode 14.6% at four to five TFLOPS, which is
nowhere near a Metal GPU's fp32 rate. What a decode step reads, on tiny.en:

| read, per step | bytes before Level 1 | bytes today | bytes if fp16 |
| --- | --- | --- | --- |
| logits weight, `embed_tokens` packed | 40 MB | 40 MB | 40 MB |
| the layers' projection weights — self q/k/v/out, cross q/out, fc1, fc2, over four layers | 33 MB | 16.5 MB | 16.5 MB |
| cross-attention keys and values, 1500 x 384 x 2 x 4 layers | 18.4 MB | 18.4 MB | 9.2 MB |
| self-attention cache, up to 448 rows | under 3 MB | under 3 MB | half |

Halving the second row was free; halving the third is not — the cache is
projected from the encoder's rows at run time, so its values are not fp16 to
begin with and storing them narrowed is a real approximation. That line is
where Level 1 ends and Level 2 begins.

fp16 arithmetic on fp32 bytes leaves every row of that table where it is.
That is the whole reason Level 3 comes last, and why it does not replace the
two above it: a half type in the EDSL changes how a kernel reads a packed
buffer, not which bytes the loader put on the device.

## Level 1: fp16 weights — done

No eacp change, no accuracy change, and it unblocked the larger repos. The
two halves landed together, written in parallel and merged.

### Pack tiny.en's projections

What the logits weight already did, generalised. `Whisper::setPacksWeights`
is a policy over every projection weight — one switch, on by default — as
`WeightPacking::ExactHalf` in `Model/TensorLoader.h`, handed to
`EncoderWeights` and `DecoderWeights` and by them to every
`loadProjectionWeight`, which goes through `SafeTensors::makeExactHalfBuffer`
and falls back to the float upload where narrowing would lose a bit. The
packed logits copy of `embed_tokens` is the same policy's doing, and
`DecoderWeights::LogitsWeight` is gone. OpenAI's checkpoints are fp16 and
HuggingFace's conversion only widens them, so for a Whisper repo the fallback
never fires and the arithmetic is bit identical: the same values, read through
`readHalf` instead of a subscript, accumulated in float32 either way.

The encoder's 28 MB of projection weights halved along with the decoder's.
Nothing else moved: the convolutions, the norms, the biases and the two
positional tables are float whatever is asked for, and `Embed`'s token table
stays the float one the gather subscripts.

### Widen rather than refuse

`loadFloatTensor` uploads a float copy of an F16 tensor instead of throwing,
through `SafeTensors::makeFloatBuffer` — F32 straight from the blob, every
other type through the `readFloats` path that already widened BF16 and F64.
`rejectPackedHalf` is gone with the refusal: a packed buffer still never
reaches a float subscript, but by construction now. The tensors are small:
tiny.en's norms, biases and convolutions together are about 2.5 MB, and the
two positional embeddings another 3 MB.

`Embed.h` was the one to look at twice, and it needed nothing. `embed_tokens`
is the gather's table and the logits weight, so an F16 repo gives the gather a
widened copy and the logits the packed original — the fp16 copy with the copy
direction reversed — and `makeExactHalfBuffer` already returned the packed
upload unchanged for an F16 tensor.

### Tests

- `Tests/Encoder/Common.h` and `Tests/Decoder/Common.h` write a third
  storage, `ProjectionStorage::EveryTensorHalf`, and the all-F16 file runs
  against the double-precision reference at the same 5e-6 as the packed
  projections: worst errors 2.85e-7 on the encoder and 1.93e-7 on the decoder,
  against 1.56e-7 and 3.11e-7 for the float file. The old refusal tests became
  tests that the tensor loads as floats holding exactly what `readFloats`
  widens to.
- `Tests/Encoder` and `Tests/Decoder` assert that under `ExactHalf` every one
  of tiny.en's projection weights came out packed.
- `Tests/Whisper/PackedWeightsTests.cpp` loads tiny.en with packing on and
  off: jfk.wav agrees token for token, and one decode step's 103,728 logits
  agree bit for bit.
- The oracle tests did not move.
- 316 tests, all passing, oracle included.

### What it was worth

The Release benchmark, 30 timed runs of jfk.wav, A B B A against the tree
before it, eacp at develop, M5 Max. The table now prints at a resolution that
can show a change of this size, which it could not before.

| | before | after | before | after |
| --- | --- | --- | --- | --- |
| transcribe, median | 14.92 ms | 14.23 ms | 14.28 ms | 14.78 ms |
| encode, median | 4.91 ms | 4.95 ms | 4.96 ms | 4.91 ms |
| decode, median | 9.81 ms | 9.05 ms | 9.11 ms | 9.67 ms |
| decode per step, median | 393 us | 362 us | 364 us | 387 us |

About 27 us off a step — 7% — for 16.5 MB less read, which at this GPU's
bandwidth is about what 16.5 MB costs, so the step is still where the bytes
say it is. **The encode did not move at all.** Its 28 MB of weights are read
once per layer against 1500 rows, so the weight read was never what the
encoder was paying for; the SIMD-group products sit at four to five TFLOPS on
the scores and the apply, and the [1500, 1500] scores are the bytes that
matter there — which is Level 2's first table row, and the one expected to
be handed back.

whisper.cpp's own columns did not move, and every transcript stayed identical
to both of them at every run.

## Level 2: fp16 activations and caches

eacp has the primitives — `writeHalf2` on the store, `readHalf2` on the load —
so a kernel can keep an intermediate packed without a half type. What it costs
is the fp32 numerics the tree has kept so far, so every candidate is measured
against the reference before it is kept, and the residual stream is the last
thing to narrow.

### Candidates, by bytes

| tensor | who writes, who reads | bytes | note |
| --- | --- | --- | --- |
| encoder attention scores, [1500, 1500] per head | the scores product; the softmax-folding apply | 54 MB written and read per layer, 432 MB an encode | the largest activation in the tree by far, and the most sensitive: a half-rounded score is an error in an exponent |
| cross-attention keys and values | `beginSequence`; every step's cross attention | 18.4 MB read a step | the third round's candidate; projected at run time, so genuinely narrowed |
| self-attention cache | each step's append; each step's attention | under 3 MB | small; do it with the row above since the reader is the same kernel |
| encoder residual stream, [1500, 384] | every layer | 2.3 MB a tensor | leave in fp32; it is what the layer norms read, and the saving is small |
| encoder output, into the decoder | the encoder; `beginSequence` | 2.3 MB | once a run; not worth its own path |

### What has to change

- An output-storage parameter on the kernels that write a candidate,
  mirroring `WeightStorage`: the scores product's store, the cross projection's
  store, the cache append in `Decoder::encodeAttentionOverCache`.
- The matching reads: the apply's A operand, `SingleQueryAttention`'s keys and
  values, `Attention.h`'s keys and values.
- Buffer sizing. `floatBytes` in `Encoder.cpp`, `Decoder.cpp` and `Whisper.cpp`
  becomes a per-buffer choice, and a packed buffer of an odd element count is
  padded to a whole word the way `uploadPackedHalves` already pads a weight.
- The SIMD product stages its operands into a `Shared<Float>` tile before
  `simdMatrix` loads them, so a packed operand is widened at staging with no
  change to the fragments. That is also the cost: the staging read is where a
  Level 3 half tile would come from instead.

### Tests

This is where tolerances move, and each move is measured rather than assumed.
A packed cache against the double-precision reference is an fp16 comparison —
about 1e-3 relative — and the oracle tests need their residuals re-measured
against ggml's own fp16 KV cache, which whisper.cpp keeps by default. The
scores product is the one to test alone first: the error in a probability
after a half-rounded score is `exp` of the rounding, and whether the transcript
survives it is an experiment, not a derivation.

### Done means

Each candidate is taken one at a time, A B B A against the build before it as
the third and fourth rounds did, and kept only where the transcript is
unchanged against both whisper.cpp columns and the benchmark row moved. The
cross-attention cache is the one expected to pay; the scores are the one
expected to be handed back.

## Level 3: fp16 arithmetic

Only if the benchmark after Levels 1 and 2 says the ALU is the limit, and it
starts in eacp:

- A `Half`, `Half2`, `Half4` family in `ValueType`, with the conversions to
  and from `Float`, so a kernel can hold a half in a register rather than
  widening on every read.
- `SimdMatrix` over `simdgroup_half8x8`, as an element choice on the fragment,
  and HLSL's `min16float` or native fp16 on D3D12, where the lane-spread
  fallback does the arithmetic 32 times over today and a half type does not
  help it.
- A half `Shared<>` tile, so the SIMD product's staging area holds what the
  fragment loads.

Half accumulation in the products gives up the bit-identical logits Level 1
keeps, which is why it is a separate decision from half storage, and why the
accumulators stay `simdgroup_float8x8` even with half operands unless a
measurement says otherwise. Per this tree's rule, all of the above is eacp
work: a gap goes upstream, not into a workaround here.

## Order, and how it is measured

1. ~~Level 1, in the two halves above, packing first. Every test in the tree
   should pass unchanged after it.~~ Done; every test passed, and the four
   new ones with them.
2. ~~The Release benchmark, 30 timed runs of jfk.wav, before and after, from a
   tree with eacp at develop.~~ Done; the table above.
3. Level 2's cross-attention cache, measured alone. Then the scores, measured
   alone, expecting to hand them back.
4. Level 3 only on the numbers, and as an eacp change first.

Every row a Release tree of its own, since a benchmark of a Debug build
measures the Debug build; and a transcript compared token for token against
both whisper.cpp columns at every step, since a faster wrong answer is not a
result.
