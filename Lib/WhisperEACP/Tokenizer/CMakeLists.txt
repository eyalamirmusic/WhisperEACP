add_library(whisper-tokenizer STATIC)

target_link_libraries(whisper-tokenizer PUBLIC whisper-core)

target_sources(whisper-tokenizer PRIVATE
        ByteEncoding.cpp
        PreTokenizer.cpp
        SpecialTokens.cpp
        Tokenizer.cpp
        Unicode.cpp)

whisper_set_default_target_setting(whisper-tokenizer)
whisper_enable_unity_build(whisper-tokenizer)
whisper_add_ide_sources(whisper-tokenizer)
