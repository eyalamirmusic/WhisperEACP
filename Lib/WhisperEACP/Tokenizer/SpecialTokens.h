#pragma once

#include <WhisperEACP/Core/Core.h>

#include <functional>
#include <map>
#include <string>
#include <string_view>

namespace WSP
{
using TokenId = int;

inline constexpr auto invalidTokenId = TokenId {-1};

// The tokens that steer decoding, named rather than numbered. Every field is
// read out of the model's tokenizer.json — the ids differ between the English
// and the multilingual vocabularies, and so does which of them exist at all.
struct SpecialTokens
{
    TokenId endOfText = invalidTokenId;
    TokenId startOfTranscript = invalidTokenId;
    TokenId startOfPrevious = invalidTokenId;
    TokenId startOfLanguageModel = invalidTokenId;
    TokenId translate = invalidTokenId;
    TokenId transcribe = invalidTokenId;
    TokenId noTimestamps = invalidTokenId;
    TokenId noSpeech = invalidTokenId;

    TokenId firstTimestamp = invalidTokenId;
    int timestampCount = 0;
    float timestampInterval = 0.0f;

    std::map<std::string, TokenId, std::less<>> languages;

    bool isTimestamp(TokenId token) const;
    TokenId timestampForSeconds(float seconds) const;
    float secondsForTimestamp(TokenId token) const;

    TokenId forLanguage(std::string_view code) const;
    std::string_view languageOf(TokenId token) const;
};

// Sorts the `<|...|>` contents of a tokenizer.json's added_tokens into the
// fields above. A two-or-three-letter code is a language, `<|12.34|>` is a
// timestamp, and the rest are matched by name.
class SpecialTokensBuilder
{
public:
    void add(std::string_view content, TokenId token);
    SpecialTokens build() const;

private:
    SpecialTokens tokens;
    float firstTimestampSeconds = 0.0f;
    float lastTimestampSeconds = 0.0f;
    TokenId lastTimestamp = invalidTokenId;
};
} // namespace WSP
