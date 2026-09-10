# Plan: F16

What it takes to run the model in fp16, split into the three things that word
can mean, in the order they pay. The earlier rounds' history — what came in
which order, and the gaps each surfaced in eacp, Miro and ResEmbed — left the
tree with the readme; this is the plan for what comes next.

## Where it stands

**fp16 weight storage is built, and one tensor uses it.** Every projection
weight has a packed-half path: `SafeTensors::makeBuffer` uploads an F16 tensor
as it lies in the blob, each linear program exists in a float-reading and a
half-reading form, and `Encoder::encodeLinear` and `Decoder::encodeLinear` pick
between them from `TensorBuffer::isPackedHalf`. `Tests/Encoder` and
`Tests/Decoder` run the packed form against the double-precision reference.

But tiny.en ships F32, and `TensorLoader::loadProjectionWeight` uploads what
the file holds. So the only packed tensor in an actual run is the fp16 copy of
`embed_tokens` that `DecoderWeights::LogitsWeight::PackedHalfCopy` builds for
the logits projection. Every other weight, every activation, both KV caches
and the cross-attention keys and values are fp32 in flight.

**An fp16 repo does not load.** `TensorLoader::loadFloatTensor` refuses a
packed tensor bound to a program with no half read — the layer norms, every
bias, both convolutions and both embedding tables. That was the right rule
against a silent bind at half the stride; it also means `whisper-base` and
everything larger, which HuggingFace ships in fp16, fail on the first norm
weight.

**eacp has half storage and no half type.** `InputBuffer::readHalf`,
`readHalf2` and `ComputeProgram::writeHalf2` move packed halves through a
buffer; `ValueType` stops at `Float`, and `SimdMatrix` loads from a
`Shared<Float>` or a float buffer and emits `simdgroup_float8x8` only.

## Why storage, and not arithmetic

The runtime is bound by bytes, not multiplies. The rounds that got the step
from 590 us to 492 us did it by halving reads — the logits weight from 80 MB
to 40 MB was the largest single win a step has had — and the fourth round's
SIMD-group products bought the encode 14.6% at four to five TFLOPS, which is
nowhere near a Metal GPU's fp32 rate. What a decode step reads, on tiny.en:

| read, per step | bytes today | bytes if fp16 |
| --- | --- | --- |
| logits weight, `embed_tokens` packed | 40 MB | 40 MB (already) |
| the layers' projection weights — self q/k/v/out, cross q/out, fc1, fc2, over four layers | 33 MB | 16.5 MB |
| cross-attention keys and values, 1500 x 384 x 2 x 4 layers | 18.4 MB | 9.2 MB |
| self-attention cache, up to 448 rows | under 3 MB | half |

Halving the second row is free; halving the third is not — the cache is
projected from the encoder's rows at run time, so its values are not fp16 to
begin with and storing them narrowed is a real approximation. That line is
where Level 1 ends and Level 2 begins.

fp16 arithmetic on fp32 bytes leaves every row of that table where it is.
That is the whole reason Level 3 comes last, and why it does not replace the
two above it: a half type in the EDSL changes how a kernel reads a packed
buffer, not which bytes the loader put on the device.

## Level 1: fp16 weights

No eacp change, no accuracy change, and it unblocks the larger repos.

### Pack tiny.en's projections

Generalise what the logits weight already does. `Whisper::setPacksLogitsWeight`
becomes a policy over every projection weight — one switch, on by default —
and `loadProjectionWeight` goes through `SafeTensors::makeExactHalfBuffer` when
it is set, falling back to the float upload where narrowing would lose a bit.
OpenAI's checkpoints are fp16 and HuggingFace's conversion only widens them,
so for a Whisper repo the fallback never fires and the arithmetic is bit
identical: the same values, read through `readHalf` instead of a subscript,
accumulated in float32 either way. `Tests/Model` already asserts the
round-trip over all 37,760,256 values.

The encoder's 28 MB of projection weights halve along with the decoder's, and
`Tests/Decoder/TinyEnDecoderTests.cpp`'s bit-identical logits check is the
template for asserting the whole thing changed nothing: the transcript of
jfk.wav token for token, and the logits of a step bit for bit.

### Widen rather than refuse

`loadFloatTensor` uploads a float copy of an F16 tensor instead of throwing.
`SafeTensors::readFloats` already does the widening for BF16 and F64; F16 joins
it on that path when the caller needs floats. The refusal stays for what it
was for — a packed buffer must never reach a float subscript — but the answer
to "this program has no half read" becomes a widened copy rather than an
error. The tensors are small: tiny.en's norms, biases and convolutions together
are about 2.5 MB, and the two positional embeddings another 3 MB.

`Embed.h` is the one to look at twice. `embed_tokens` is the gather's table and
the logits weight, so an F16 repo gives the gather a widened copy and the
logits the packed original, which is `LogitsWeight::PackedHalfCopy` with the
copy direction reversed. `makeExactHalfBuffer` returns the packed upload
unchanged for an F16 tensor already, so that half needs nothing.

### Tests

- `Tests/Encoder/Common.h`'s `syntheticEncoderFile` writes F16 projections and
  F32 everything else under `ProjectionStorage::PackedHalf`. A third storage,
  every tensor F16, is what exercises the widening path and the gather.
- The packed-projection comparisons in `Tests/Encoder` and `Tests/Decoder`
  already run at the float32-accumulation tolerance, 5e-6. They stay there.
- `Whisper/` gets a test that loads tiny.en with packing on and off and
  compares the transcript and one step's logits bit for bit.
- The oracle tests do not move: their reference is ggml's fp16 weights through
  its own CPU path, and this brings ours closer to that, not further.

### Done means

A run of tiny.en reads 16.5 MB less a step and produces the same bytes;
`whisper-base` loads and transcribes; the Release benchmark says what the
saving was worth on the step and on the encode.

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

1. Level 1, in the two halves above, packing first. Every test in the tree
   should pass unchanged after it.
2. The Release benchmark, 30 timed runs of jfk.wav, before and after, from a
   tree with eacp at develop.
3. Level 2's cross-attention cache, measured alone. Then the scores, measured
   alone, expecting to hand them back.
4. Level 3 only on the numbers, and as an eacp change first.

Every row a Release tree of its own, since a benchmark of a Debug build
measures the Debug build; and a transcript compared token for token against
both whisper.cpp columns at every step, since a faster wrong answer is not a
result.
