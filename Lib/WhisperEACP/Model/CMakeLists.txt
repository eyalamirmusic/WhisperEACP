add_library(whisper-model STATIC)

# Miro rides along on eacp-core, which whisper-core links through eacp-gpu, so
# <Miro/Json.h> needs no dependency of its own here.
#
# eacp-network is for ModelFetch: the model is a 151 MB download, and this is
# the module that knows which four files it is. PUBLIC because ModelFetch.h is
# written in eacp::OnlineResource's own types — the Info list, the freshness and
# the per-transfer progress are all its, and a caller that shows a download
# reads them.
target_link_libraries(whisper-model PUBLIC whisper-core eacp-network)

target_sources(whisper-model PRIVATE
        ModelConfig.cpp
        ModelFetch.cpp
        ModelIO.cpp
        PreprocessorConfig.cpp
        SafeTensors.cpp
        TensorLoader.cpp
        TensorType.cpp)

whisper_set_default_target_setting(whisper-model)
whisper_enable_unity_build(whisper-model)
whisper_add_ide_sources(whisper-model)
