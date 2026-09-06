#pragma once

// Whisper's byte-level BPE, read from the vocabulary and merges in the model's
// tokenizer.json, together with the special tokens that steer decoding.

#include <WhisperEACP/Core/Core.h>
#include <WhisperEACP/Tokenizer/ByteEncoding.h>
#include <WhisperEACP/Tokenizer/PreTokenizer.h>
#include <WhisperEACP/Tokenizer/SpecialTokens.h>
#include <WhisperEACP/Tokenizer/Unicode.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace WSP
{
class TokenizerError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class Tokenizer
{
public:
    static Tokenizer fromJsonText(std::string_view jsonText);
    static Tokenizer fromFile(const std::filesystem::path& path);

    Vector<TokenId> encode(std::string_view text) const;

    std::string decode(Span<const TokenId> tokens) const;
    std::string decodeKeepingSpecialTokens(Span<const TokenId> tokens) const;

    int size() const;
    bool isSpecial(TokenId token) const;

    // The vocabulary's own spelling of a token, still in the byte-level
    // alphabet: `Ġthe`, not ` the`.
    std::string_view textForToken(TokenId token) const;
    TokenId tokenForText(std::string_view tokenText) const;

    const SpecialTokens& specials() const { return specialTokens; }

private:
    // Transparent, so a std::string_view looks a token up without first being
    // copied into a std::string.
    struct StringHash
    {
        using is_transparent = void;

        std::size_t operator()(std::string_view text) const
        {
            return std::hash<std::string_view> {}(text);
        }
    };

    using TokenIdByText =
        std::unordered_map<std::string, TokenId, StringHash, std::equal_to<>>;

    struct Merge
    {
        int rank = 0;
        TokenId merged = invalidTokenId;
    };

    void encodePreToken(std::string_view preToken, Vector<TokenId>& tokens) const;
    TokenId specialTokenAt(std::string_view text, std::size_t position) const;
    std::string decodeTokens(Span<const TokenId> tokens, bool keepSpecial) const;

    Vector<std::string> tokenTexts;
    Vector<std::uint8_t> specialFlags;
    TokenIdByText tokensByText;
    std::unordered_map<std::uint64_t, Merge> merges;
    SpecialTokens specialTokens;
};
} // namespace WSP
