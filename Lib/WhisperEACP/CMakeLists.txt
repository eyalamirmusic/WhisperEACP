add_subdirectory(Core)

# Above Core, below everything else: the mel front-end and the encoder are
# written against the format this module fixes, and it is the only place that
# talks to a capture device.
add_subdirectory(Audio)

# The op set the encoder and decoder are assembled from. Nothing below it
# depends on it, so it names no Whisper shape of its own.
add_subdirectory(Kernels)

# The front-end, on top of Audio's contract.
add_subdirectory(Mel)

# The weights and the vocabulary. Both read published files rather than a
# format of ours, and neither includes a header from the two above.
add_subdirectory(Model)
add_subdirectory(Tokenizer)

# The first module above the seam: it includes headers from two of the four
# below it, which is what "assembled from Kernels/, fed by Mel/ and Model/"
# means once there is something to assemble.
add_subdirectory(Encoder)

# Beside the encoder rather than above it: the encoder's output reaches the
# decoder as a buffer, so neither includes a header from the other.
add_subdirectory(Decoder)

# The top of the tree, and the only module that includes a header from every
# other one: samples in, a transcript out, with the greedy search the decoder
# deliberately left to a layer that could hold the generation config.
add_subdirectory(Whisper)

# The whole library, for consumers that don't want to name the modules they use.
add_library(WhisperEACP INTERFACE)
target_link_libraries(WhisperEACP INTERFACE
        whisper-core
        whisper-audio
        whisper-kernels
        whisper-mel
        whisper-model
        whisper-tokenizer
        whisper-encoder
        whisper-decoder
        whisper-runtime)
