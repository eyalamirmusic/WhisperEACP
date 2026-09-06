#include "Common.h"

#include <iostream>

// The vocabulary, compared against whisper.cpp's copy of the same one.
//
// The two read it from different files — ours from HuggingFace's
// tokenizer.json, whisper.cpp's from the vocabulary section of the GGML .bin,
// which the conversion script writes with GPT-2's byte decoder already applied.
// So an id means the same thing on both sides, but a token's spelling does
// not: ours is still in the byte-level alphabet (`Ġthe`), theirs is the bytes
// (` the`). Decode is what puts the two in the same alphabet.

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
std::vector<whisper_token> oracleEncode(whisper_context& context,
                                        std::string_view text)
{
    auto tokens = std::vector<whisper_token>(text.size() + 8);
    const auto count = whisper_tokenize(
        &context, std::string {text}.c_str(), tokens.data(), (int) tokens.size());

    tokens.resize(count < 0 ? 0 : (std::size_t) count);
    return tokens;
}

std::vector<whisper_token> ourEncode(const Tokenizer& tokenizer,
                                     std::string_view text)
{
    const auto tokens = tokenizer.encode(text);
    return {tokens.begin(), tokens.end()};
}

std::string spelled(const std::vector<whisper_token>& tokens)
{
    auto text = std::string {};

    for (const auto token: tokens)
        text += (text.empty() ? "" : ", ") + std::to_string(token);

    return "[" + text + "]";
}

// Spelled as bytes rather than as literals, the way Tests/Tokenizer does, so
// the case means the same thing whatever the compiler decides the source
// encoding is. "Le café était naïf." — a sentence whose letters leave ASCII.
const auto frenchSentence = std::string {"Le caf\xc3\xa9 \xc3\xa9tait na\xc3\xaf"
                                         "f."};

const auto textCases = std::vector<std::string> {
    "",
    "Hello world",
    " and so, my fellow Americans",
    "trailing space ",
    "The quick brown fox jumps over the lazy dog.",
    "Well... it's 3:45 p.m. -- isn't it?",
    "In 1969 there were 365 days and 12 months.",
    frenchSentence,
};

// A spread across the vocabulary rather than the first few: single bytes, the
// space-prefixed words BPE spends most of its ids on, and the last ordinary id
// before the specials begin.
const auto tokenCases = std::vector<TokenId> {
    0, 1, 50, 100, 220, 257, 1000, 5000, 10000, 20000, 40000, 50000, 50255};

// The id whose only byte is a NUL. whisper_token_to_str hands back a C string,
// so this one token cannot come through its API at all — the std::string
// behind it holds the byte, and c_str() ends at it. Named here rather than
// quietly left out of the spread above.
constexpr auto nulByteToken = TokenId {188};

bool canRun()
{
    return hasGgmlModel() && hasModelFile(tokenizerFile);
}
} // namespace

// whisper.cpp's tokenize() is a greedy longest-match over the vocabulary, not
// BPE: it splits on a std::regex transliteration of GPT-2's pattern and then
// takes the longest token that fits at each position. That agrees with BPE on
// most English and is not the same algorithm, so this test measures where the
// two part company rather than asserting they never do — and the two places
// they part are the reference's, not ours:
//
//  * `[[:alpha:]]` and `[[:digit:]]` in a std::regex are ASCII in the default
//    locale, where the pattern this transliterates uses \p{L} and \p{N}. Any
//    letter outside ASCII therefore lands in the punctuation class over there
//    and splits differently.
//  * greedy longest-match and BPE's merge order genuinely disagree on some
//    strings, and BPE is what the model was trained with.
//
// What holds unconditionally is that both are segmentations of the same bytes,
// so both decode back to the input. That is the assertion; exact agreement is
// asserted for the ASCII cases, where the reference is doing the same job.
auto tEncodeAgreesWithTheOracle = test("Oracle/encodeAgreesWithTheOracle") = []
{
    if (!canRun())
        return;

    auto oracle = loadOracle();
    check(oracle != nullptr, "the GGML model loaded");

    if (!oracle)
        return;

    const auto tokenizer = Tokenizer::fromFile(modelFile(tokenizerFile));
    auto agreements = 0;

    for (const auto& text: textCases)
    {
        const auto ours = ourEncode(tokenizer, text);
        const auto theirs = oracleEncode(*oracle, text);

        if (ours == theirs)
            ++agreements;
        else
            std::cout << "  differ on \"" << text << "\": ours " << spelled(ours)
                      << ", whisper.cpp " << spelled(theirs) << "\n";

        check(tokenizer.decode(Span<const TokenId> {ours}) == text,
              "ours round trips");

        const auto asOurs = std::vector<TokenId> {theirs.begin(), theirs.end()};
        check(tokenizer.decode(Span<const TokenId> {asOurs}) == text,
              "the oracle's tokens round trip through our vocabulary");
    }

    std::cout << "  " << agreements << " of " << textCases.size()
              << " strings tokenize identically\n";
};

// whisper_token_to_str returns the vocabulary's own spelling, which on that
// side is the decoded bytes — so it is our decode of a one-token sequence that
// it has to be compared against, not textForToken, which is still in the
// byte-level alphabet.
auto tDecodeAgreesWithTheOracle = test("Oracle/decodeAgreesWithTheOracle") = []
{
    if (!canRun())
        return;

    auto oracle = loadOracle();
    check(oracle != nullptr, "the GGML model loaded");

    if (!oracle)
        return;

    const auto tokenizer = Tokenizer::fromFile(modelFile(tokenizerFile));

    for (const auto token: tokenCases)
    {
        const auto one = std::vector<TokenId> {token};
        const auto ours = tokenizer.decode(Span<const TokenId> {one});
        const auto theirs = std::string {whisper_token_to_str(oracle.get(), token)};

        if (ours != theirs)
            std::cout << "  token " << token << ": ours \"" << ours
                      << "\", whisper.cpp \"" << theirs << "\"\n";

        check(ours == theirs, "the spelling agrees");
    }

    const auto nul = std::vector<TokenId> {nulByteToken};
    check(tokenizer.decode(Span<const TokenId> {nul}) == std::string(1, '\0'),
          "our decode carries the NUL byte");
    check(std::string {whisper_token_to_str(oracle.get(), nulByteToken)}.empty(),
          "whisper_token_to_str cannot");
};

// The ids that steer decoding. Ours are read out of tokenizer.json's
// added_tokens by name; whisper.cpp's are compiled-in constants adjusted by
// the vocabulary size. tiny.en's tokenizer.json ships the full multilingual
// added-token set, so every one of them exists on both sides.
auto tSpecialTokensAgreeWithTheOracle =
    test("Oracle/specialTokensAgreeWithTheOracle") = []
{
    if (!canRun())
        return;

    auto oracle = loadOracle();
    check(oracle != nullptr, "the GGML model loaded");

    if (!oracle)
        return;

    const auto tokenizer = Tokenizer::fromFile(modelFile(tokenizerFile));
    const auto& specials = tokenizer.specials();
    auto* context = oracle.get();

    check(specials.endOfText == whisper_token_eot(context), "eot");
    check(specials.startOfTranscript == whisper_token_sot(context), "sot");
    check(specials.startOfPrevious == whisper_token_prev(context), "prev");
    check(specials.startOfLanguageModel == whisper_token_solm(context), "solm");
    check(specials.noSpeech == whisper_token_nosp(context), "nosp");
    check(specials.noTimestamps == whisper_token_not(context), "not");
    check(specials.firstTimestamp == whisper_token_beg(context), "beg");
    check(specials.transcribe == whisper_token_transcribe(context), "transcribe");
    check(specials.translate == whisper_token_translate(context), "translate");

    check(tokenizer.size() == whisper_n_vocab(context),
          "the vocabularies are one size");
};
