#include <WhisperEACP/Tokenizer/Tokenizer.h>

#include <Miro/Json.h>
#include <Miro/Unicode.h>
#include <NanoTest/NanoTest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <string_view>

using namespace nano;
using namespace WSP;

namespace
{
// Spelled as bytes rather than as literals so the tests mean the same thing
// whatever the compiler decides the source encoding is.
constexpr auto eAcute = std::string_view {"\xc3\xa9"};
constexpr auto iDiaeresis = std::string_view {"\xc3\xaf"};
constexpr auto combiningAcute = std::string_view {"\xcc\x81"};
constexpr auto noBreakSpace = std::string_view {"\xc2\xa0"};
constexpr auto zeroWidthSpace = std::string_view {"\xe2\x80\x8b"};
constexpr auto ideographicSpace = std::string_view {"\xe3\x80\x80"};
constexpr auto ideographicFullStop = std::string_view {"\xe3\x80\x82"};
constexpr auto niHaoShiJie =
    std::string_view {"\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c"};
constexpr auto grinningFace = std::string_view {"\xf0\x9f\x98\x80"};
constexpr auto thumbsUp = std::string_view {"\xf0\x9f\x91\x8d"};

std::string joined(std::initializer_list<std::string_view> parts)
{
    auto text = std::string {};

    for (auto part: parts)
        text += part;

    return text;
}

bool encodesTo(const Tokenizer& tokenizer,
               std::string_view text,
               std::initializer_list<TokenId> expected)
{
    const auto tokens = tokenizer.encode(text);
    return std::equal(
        tokens.begin(), tokens.end(), expected.begin(), expected.end());
}

bool roundTrips(const Tokenizer& tokenizer, std::string_view text)
{
    const auto tokens = tokenizer.encode(text);
    return tokenizer.decode(tokens) == text;
}

bool splitsInto(std::string_view text,
                std::initializer_list<std::string_view> expected)
{
    const auto preTokens = PreTokenizer::split(text);
    return std::equal(
        preTokens.begin(), preTokens.end(), expected.begin(), expected.end());
}

// The mapping Unicode.cpp is written to be: our three classes are Miro's
// general categories restated, and nothing else.
Unicode::Category miroCategoryOf(char32_t codePoint)
{
    if (Miro::Unicode::isWhitespace(codePoint))
        return Unicode::Category::whitespace;

    if (Miro::Unicode::isLetter(codePoint))
        return Unicode::Category::letter;

    if (Miro::Unicode::isNumber(codePoint))
        return Unicode::Category::number;

    return Unicode::Category::other;
}

std::filesystem::path fixturePath()
{
    return std::filesystem::path {WHISPER_TOKENIZER_FIXTURE_DIR}
           / "MiniTokenizer.json";
}

const Tokenizer& fixtureTokenizer()
{
    static const auto tokenizer = Tokenizer::fromFile(fixturePath());
    return tokenizer;
}

// The real vocabulary is a download, so every test that wants one returns
// early when none was pointed at, exactly as the GPU tests do without a
// device.
//
// Four places, most specific first. The two tokenizer-only ones came first and
// are kept because they name a file rather than a directory, which is what a
// checkout that is not a whole HF repo has. The two model-directory ones were
// added once tokenizer.json joined the fetch: the file is downloaded beside the
// other three now, so a build configured with -DWHISPER_EACP_FETCH_MODEL=ON runs
// these without being told where anything is.
std::filesystem::path realTokenizerPath()
{
    const auto exists = [](const std::filesystem::path& path)
    {
        auto error = std::error_code {};
        return !path.empty() && std::filesystem::is_regular_file(path, error);
    };

    if (const auto* fromEnvironment = std::getenv("WHISPER_TOKENIZER_JSON"))
        return fromEnvironment;

    if (exists(WHISPER_TOKENIZER_JSON_PATH))
        return WHISPER_TOKENIZER_JSON_PATH;

    if (const auto* fromModelDirectory = std::getenv("WHISPER_MODEL_DIR"))
    {
        const auto inDirectory =
            std::filesystem::path {fromModelDirectory} / "tokenizer.json";

        if (exists(inDirectory))
            return inDirectory;
    }

    return std::filesystem::path {WHISPER_EACP_MODEL_DIR} / "tokenizer.json";
}

bool hasRealTokenizer()
{
    const auto path = realTokenizerPath();

    auto error = std::error_code {};
    return !path.empty() && std::filesystem::is_regular_file(path, error);
}

const Tokenizer& realTokenizer()
{
    static const auto tokenizer = Tokenizer::fromFile(realTokenizerPath());
    return tokenizer;
}
} // namespace

// --- The byte-level layer -----------------------------------------------

auto tByteEncodingIsABijection = test("Tokenizer/byteEncodingIsABijection") = []
{
    for (auto byte = 0; byte < 256; ++byte)
    {
        const auto codePoint =
            ByteEncoding::codePointForByte(static_cast<unsigned char>(byte));

        check(ByteEncoding::byteForCodePoint(codePoint) == byte);
    }
};

// The bytes with no printable form of their own are pushed above U+00FF, in
// byte order: a space becomes U+0120 and a newline U+010A, which is what the
// 'Ġ' and 'Ċ' in a GPT-2 vocabulary are.
auto tByteEncodingLifesUnprintableBytes =
    test("Tokenizer/byteEncodingLiftsUnprintableBytes") = []
{
    check(ByteEncoding::codePointForByte(' ') == 0x120);
    check(ByteEncoding::codePointForByte('\n') == 0x10A);
    check(ByteEncoding::codePointForByte(0) == 0x100);
    check(ByteEncoding::codePointForByte(0x7F) == 0x121);
    check(ByteEncoding::codePointForByte(0xAD) == 0x143);

    check(ByteEncoding::codePointForByte('a') == U'a');
    check(ByteEncoding::codePointForByte(0xFF) == 0xFF);
};

auto tByteEncodingRoundTripsEveryByte =
    test("Tokenizer/byteEncodingRoundTripsEveryByte") = []
{
    auto allBytes = std::string {};

    for (auto byte = 0; byte < 256; ++byte)
        allBytes += static_cast<char>(byte);

    const auto printable = ByteEncoding::toPrintable(allBytes);
    check(ByteEncoding::fromPrintable(printable) == allBytes);

    check(ByteEncoding::toPrintable(" the") == "\xc4\xa0the");
};

auto tByteEncodingIsInvisibleOnAscii =
    test("Tokenizer/byteEncodingIsInvisibleOnAscii") = []
{ check(ByteEncoding::toPrintable("Hello!") == "Hello!"); };

// --- Unicode classification ---------------------------------------------

auto tUnicodeCategories = test("Tokenizer/unicodeCategories") = []
{
    using Unicode::Category;
    using Unicode::categoryOf;

    check(categoryOf(U'a') == Category::letter);
    check(categoryOf(U'7') == Category::number);
    check(categoryOf(U'!') == Category::other);
    check(categoryOf(U' ') == Category::whitespace);
    check(categoryOf(U'\n') == Category::whitespace);

    check(categoryOf(0x00E9) == Category::letter);
    check(categoryOf(0x4F60) == Category::letter);
    check(categoryOf(0x0301) == Category::other);
    check(categoryOf(0x1F600) == Category::other);
    check(categoryOf(0x3002) == Category::other);
    check(categoryOf(0x00B2) == Category::number);
    check(categoryOf(0x0663) == Category::number);

    check(categoryOf(0x00A0) == Category::whitespace);
    check(categoryOf(0x3000) == Category::whitespace);
    check(categoryOf(0x200B) == Category::other);
    check(categoryOf(0xFEFF) == Category::other);
};

auto tInvalidUtf8ReadsAsOneByte = test("Tokenizer/invalidUtf8ReadsAsOneByte") = []
{
    const auto lone = std::string_view {"\x80"};
    check(Unicode::readCodePoint(lone, 0).byteLength == 1);
    check(Unicode::readCodePoint(lone, 0).category == Unicode::Category::other);

    const auto truncated = std::string_view {"\xe4\xbd"};
    check(Unicode::readCodePoint(truncated, 0).byteLength == 1);

    const auto overlong = std::string_view {"\xc0\xaf"};
    check(Unicode::readCodePoint(overlong, 0).byteLength == 1);

    check(Unicode::readCodePoint(niHaoShiJie, 0).byteLength == 3);
    check(Unicode::readCodePoint(niHaoShiJie, 0).value == 0x4F60);
    check(Unicode::readCodePoint(grinningFace, 0).byteLength == 4);
    check(Unicode::readCodePoint(grinningFace, 0).value == 0x1F600);
};

// A sweep rather than a sample, because the thing that can go wrong is a
// whole run of code points changing class when Miro moves to a later Unicode
// version. Dense below U+30000, where every script a tokenizer meets lives,
// and every 17th code point above, which is enough to notice a plane move.
auto tUnicodeAgreesWithMiro = test("Tokenizer/unicodeAgreesWithMiro") = []
{
    constexpr auto denselySwept = char32_t {0x30000};
    constexpr auto strideAbove = char32_t {17};
    constexpr auto lastCodePoint = char32_t {0x10FFFF};

    auto mismatches = 0;

    for (auto codePoint = char32_t {0}; codePoint < denselySwept; ++codePoint)
        if (Unicode::categoryOf(codePoint) != miroCategoryOf(codePoint))
            ++mismatches;

    for (auto codePoint = denselySwept; codePoint <= lastCodePoint;
         codePoint += strideAbove)
        if (Unicode::categoryOf(codePoint) != miroCategoryOf(codePoint))
            ++mismatches;

    check(mismatches == 0);
};

// A rejected sequence reports its lead byte, and that byte is a byte: 0xC3
// alone must not come back as the letter U+00C3, or the byte-level layer
// would classify half of a two-byte sequence as a word character.
auto tUtf8DecodeEdges = test("Tokenizer/utf8DecodeEdges") = []
{
    const auto loneLeadByte = Unicode::readCodePoint("\xc3", 0);
    check(loneLeadByte.byteLength == 1);
    check(loneLeadByte.category == Unicode::Category::other);

    check(Unicode::readCodePoint("\xed\xa0\x80", 0).byteLength == 1);
    check(Unicode::readCodePoint("\xf4\x90\x80\x80", 0).byteLength == 1);

    const auto inside = Unicode::readCodePoint(joined({"ab", niHaoShiJie, "z"}), 2);
    check(inside.value == 0x4F60);
    check(inside.byteLength == 3);
    check(inside.category == Unicode::Category::letter);
};

// --- The pre-tokenizer --------------------------------------------------

auto tPreTokenizerSplitsWords = test("Tokenizer/preTokenizerSplitsWords") = []
{
    check(splitsInto("Hello world", {"Hello", " world"}));
    check(splitsInto(" Hello, world!", {" Hello", ",", " world", "!"}));
    check(splitsInto("Don't stop", {"Don", "'t", " stop"}));
    check(splitsInto("It's", {"It", "'s"}));
    check(splitsInto("1234 567.89", {"1234", " 567", ".", "89"}));
    check(splitsInto("", {}));
};

// `\s+(?!\S)` is the rule that leaves the last space of a run to the word
// that follows it, and it is the one a hand-written scanner gets wrong.
auto tPreTokenizerLeavesOneSpaceToTheNextWord =
    test("Tokenizer/preTokenizerLeavesOneSpaceToTheNextWord") = []
{
    check(splitsInto("  double  spaces", {" ", " double", " ", " spaces"}));
    check(splitsInto("x  ", {"x", "  "}));
    check(splitsInto("a\tb", {"a", "\t", "b"}));
    check(splitsInto("\n\nx", {"\n", "\n", "x"}));
};

auto tPreTokenizerOnNonAscii = test("Tokenizer/preTokenizerOnNonAscii") = []
{
    check(
        splitsInto(joined({"caf", eAcute, " x"}), {joined({"caf", eAcute}), " x"}));
    check(splitsInto(joined({niHaoShiJie, ideographicFullStop, "ok"}),
                     {niHaoShiJie, ideographicFullStop, "ok"}));
    check(splitsInto(joined({grinningFace, " emoji ", thumbsUp}),
                     {grinningFace, " emoji", joined({" ", thumbsUp})}));

    // A combining mark is not \p{L}, so it starts a pre-token of its own.
    check(
        splitsInto(joined({"e", combiningAcute, "x"}), {"e", combiningAcute, "x"}));

    check(splitsInto(joined({"a", noBreakSpace, "b"}), {"a", noBreakSpace, "b"}));
    check(splitsInto(joined({"  ", ideographicSpace}),
                     {joined({"  ", ideographicSpace})}));
    check(splitsInto(joined({"  ", zeroWidthSpace}),
                     {" ", joined({" ", zeroWidthSpace})}));
};

auto tPreTokenizerKeepsInvalidBytes =
    test("Tokenizer/preTokenizerKeepsInvalidBytes") = []
{
    check(splitsInto("a\x80\x81z", {"a", "\x80\x81", "z"}));
    check(splitsInto("\xff\xfe", {"\xff\xfe"}));
};

// --- The committed fixture ----------------------------------------------

auto tFixtureLoads = test("Tokenizer/fixtureLoads") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(tokenizer.size() == 272);
    check(tokenizer.tokenForText("a") == 'a');
    check(tokenizer.tokenForText("\xc4\xa0the") == 261);
    check(tokenizer.textForToken(261) == "\xc4\xa0the");
    check(tokenizer.tokenForText("nonesuch") == invalidTokenId);
};

// The lower-rank merge has to win: "b c" is rank 0 and "a b" rank 1, so "abc"
// is a + bc and never ab + c, which is what a left-to-right sweep would give.
auto tFixtureAppliesMergesInRankOrder =
    test("Tokenizer/fixtureAppliesMergesInRankOrder") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(encodesTo(tokenizer, "abc", {'a', 256}));
    check(encodesTo(tokenizer, "ab", {257}));
    check(encodesTo(tokenizer, "cab", {'c', 257}));
    check(encodesTo(tokenizer, "abcabc", {'a', 256, 'a', 256}));
};

// "Ġthe" only exists after "h e", then "Ġ t", then "Ġt he" have each fired.
auto tFixtureAppliesMergesRepeatedly =
    test("Tokenizer/fixtureAppliesMergesRepeatedly") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(encodesTo(tokenizer, "the", {260}));
    check(encodesTo(tokenizer, " the", {261}));
    check(encodesTo(tokenizer, "  the ", {' ', 261, ' '}));
    check(encodesTo(tokenizer, "he", {258}));
};

auto tFixtureEncodesEmptyInput = test("Tokenizer/fixtureEncodesEmptyInput") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(encodesTo(tokenizer, "", {}));
    check(roundTrips(tokenizer, ""));
};

auto tFixtureRoundTripsBytes = test("Tokenizer/fixtureRoundTripsBytes") = []
{
    const auto& tokenizer = fixtureTokenizer();

    check(encodesTo(tokenizer, eAcute, {262}));
    check(encodesTo(tokenizer, joined({"the", eAcute}), {260, 262}));

    check(roundTrips(tokenizer, "the quick brown fox"));
    check(roundTrips(tokenizer, joined({"caf", eAcute, " ", niHaoShiJie})));
    check(roundTrips(tokenizer, joined({grinningFace, " ", thumbsUp})));

    // Bytes that are not valid UTF-8 on their own still survive the trip: the
    // byte-level layer never looks at them as text.
    check(encodesTo(tokenizer, "\xff\xfe", {0xff, 0xfe}));
    check(encodesTo(tokenizer, "a\x80z", {'a', 0x80, 'z'}));
    check(roundTrips(tokenizer, "\xff\xfe\x80\x01"));

    auto everyByte = std::string {};

    for (auto byte = 1; byte < 256; ++byte)
        everyByte += static_cast<char>(byte);

    check(roundTrips(tokenizer, everyByte));
};

auto tFixtureSpecialTokens = test("Tokenizer/fixtureSpecialTokens") = []
{
    const auto& specials = fixtureTokenizer().specials();

    check(specials.endOfText == 263);
    check(specials.startOfTranscript == 264);
    check(specials.translate == 266);
    check(specials.transcribe == 267);
    check(specials.noTimestamps == 268);
    check(specials.forLanguage("en") == 265);
    check(specials.forLanguage("de") == invalidTokenId);
    check(specials.languageOf(265) == "en");

    check(specials.firstTimestamp == 269);
    check(specials.timestampCount == 3);
    check(specials.isTimestamp(270));
    check(!specials.isTimestamp(268));
    check(specials.secondsForTimestamp(271) > 0.039f);
    check(specials.secondsForTimestamp(271) < 0.041f);
    check(specials.timestampForSeconds(0.02f) == 270);
};

auto tFixtureSpecialTokensInText = test("Tokenizer/fixtureSpecialTokensInText") = []
{
    const auto& tokenizer = fixtureTokenizer();
    const auto text = std::string {"<|startoftranscript|>abc<|endoftext|>"};

    check(encodesTo(tokenizer, text, {264, 'a', 256, 263}));

    const auto tokens = tokenizer.encode(text);
    check(tokenizer.decode(tokens) == "abc");
    check(tokenizer.decodeKeepingSpecialTokens(tokens) == text);

    check(tokenizer.isSpecial(263));
    check(!tokenizer.isSpecial('a'));

    // An unknown `<|...|>` is ordinary text, not a token.
    check(roundTrips(tokenizer, "<|nope|>"));
};

auto tMalformedJsonThrows = test("Tokenizer/malformedJsonThrows") = []
{
    auto threw = false;

    try
    {
        Tokenizer::fromJsonText("{ not json");
    }
    catch (const Miro::Json::ParseError&)
    {
        threw = true;
    }

    check(threw);

    threw = false;

    try
    {
        Tokenizer::fromJsonText(R"({"model": {"vocab": {}}})");
    }
    catch (const TokenizerError&)
    {
        threw = true;
    }

    check(threw);
};

// --- The real vocabulary, when one is available -------------------------

auto tRealVocabularyShape = test("Tokenizer/realVocabularyShape") = []
{
    if (!hasRealTokenizer())
        return;

    const auto& tokenizer = realTokenizer();

    check(tokenizer.size() == 51864);

    // A byte-level vocabulary has to spell all 256 bytes, or BPE has nothing
    // to start from.
    for (auto byte = 0; byte < 256; ++byte)
    {
        auto piece = std::string {};
        Unicode::appendUtf8(
            piece, ByteEncoding::codePointForByte(static_cast<unsigned char>(byte)));

        check(tokenizer.tokenForText(piece) != invalidTokenId);
    }
};

auto tRealEncodesLikeHuggingFace = test("Tokenizer/realEncodesLikeHuggingFace") = []
{
    if (!hasRealTokenizer())
        return;

    const auto& tokenizer = realTokenizer();

    check(encodesTo(tokenizer, "Hello world", {15496, 995}));
    check(encodesTo(tokenizer, " Hello, world!", {18435, 11, 995, 0}));
    check(encodesTo(tokenizer,
                    "The quick brown fox jumps over the lazy dog.",
                    {464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13}));
    check(encodesTo(tokenizer, "Don't stop believing", {3987, 470, 2245, 14773}));
    check(encodesTo(tokenizer, "1234 567.89", {1065, 2682, 642, 3134, 13, 4531}));
    check(encodesTo(tokenizer, "", {}));
};

auto tRealEncodesNonAsciiLikeHuggingFace =
    test("Tokenizer/realEncodesNonAsciiLikeHuggingFace") = []
{
    if (!hasRealTokenizer())
        return;

    const auto& tokenizer = realTokenizer();

    check(encodesTo(tokenizer,
                    joined({"caf", eAcute, " na", iDiaeresis, "ve"}),
                    {66, 1878, 2634, 41492}));
    check(encodesTo(
        tokenizer, niHaoShiJie, {19526, 254, 25001, 121, 10310, 244, 45911, 234}));
    check(encodesTo(tokenizer,
                    joined({grinningFace, " emoji ", thumbsUp}),
                    {47249, 222, 44805, 50169, 235}));
};

auto tRealRoundTrips = test("Tokenizer/realRoundTrips") = []
{
    if (!hasRealTokenizer())
        return;

    const auto& tokenizer = realTokenizer();

    check(roundTrips(tokenizer, "The quick brown fox jumps over the lazy dog."));
    check(roundTrips(tokenizer, joined({"caf", eAcute, " na", iDiaeresis, "ve"})));
    check(roundTrips(tokenizer, joined({niHaoShiJie, ideographicFullStop})));
    check(roundTrips(tokenizer, joined({grinningFace, " emoji ", thumbsUp})));
    check(roundTrips(tokenizer, "  double  spaces\n\nnewlines\t\ttabs"));
    check(roundTrips(tokenizer, "\xff\xfe\x80"));

    auto everyByte = std::string {};

    for (auto byte = 1; byte < 256; ++byte)
        everyByte += static_cast<char>(byte);

    check(roundTrips(tokenizer, everyByte));
};

// tiny.en is English-only, but its tokenizer.json still carries the whole
// multilingual special-token block: 99 language tokens, both task tokens, and
// 1501 timestamps at a 0.02 s step covering the 30 s window.
auto tRealSpecialTokens = test("Tokenizer/realSpecialTokens") = []
{
    if (!hasRealTokenizer())
        return;

    const auto& tokenizer = realTokenizer();
    const auto& specials = tokenizer.specials();

    check(specials.endOfText == 50256);
    check(specials.startOfTranscript == 50257);
    check(specials.translate == 50357);
    check(specials.transcribe == 50358);
    check(specials.startOfLanguageModel == 50359);
    check(specials.startOfPrevious == 50360);
    check(specials.noSpeech == 50361);
    check(specials.noTimestamps == 50362);

    check(specials.forLanguage("en") == 50258);
    check(specials.forLanguage("haw") == 50351);
    check(specials.languages.size() == 99);
    check(specials.languageOf(50258) == "en");

    check(specials.firstTimestamp == 50363);
    check(specials.timestampCount == 1501);
    check(specials.timestampInterval > 0.0199f);
    check(specials.timestampInterval < 0.0201f);
    check(specials.timestampForSeconds(30.0f) == 51863);
    check(specials.secondsForTimestamp(51863) > 29.99f);
    check(specials.secondsForTimestamp(51863) < 30.01f);

    check(tokenizer.isSpecial(50257));
    check(tokenizer.isSpecial(50363));
    check(!tokenizer.isSpecial(15496));

    const auto prompt =
        std::string {"<|startoftranscript|><|notimestamps|> Hello<|endoftext|>"};

    check(encodesTo(tokenizer, prompt, {50257, 50362, 18435, 50256}));

    const auto tokens = tokenizer.encode(prompt);
    check(tokenizer.decode(tokens) == " Hello");
    check(tokenizer.decodeKeepingSpecialTokens(tokens) == prompt);
};
