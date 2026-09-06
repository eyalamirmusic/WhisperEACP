#include "Tokenizer.h"

#include <Miro/Json.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <queue>
#include <utility>
#include <vector>

namespace WSP
{
namespace
{
using Miro::Json::Value;

constexpr auto specialTokenOpening = std::string_view {"<|"};
constexpr auto specialTokenClosing = std::string_view {"|>"};

const Value& memberOrThrow(const Value& parent, const char* key)
{
    if (!parent.isObject())
        throw TokenizerError(std::string("expected an object holding '") + key
                             + "'");

    const auto* found = Miro::Json::find(parent.asObject(), key);

    if (found == nullptr)
        throw TokenizerError(std::string("tokenizer.json has no '") + key + "'");

    return *found;
}

TokenId asTokenId(const Value& value)
{
    if (!value.isNumber())
        throw TokenizerError("expected a token id");

    return static_cast<TokenId>(value.asNumber());
}

std::pair<std::string_view, std::string_view> splitMergeRule(const Value& entry)
{
    if (entry.isArray())
    {
        const auto& pair = entry.asArray();

        if (pair.size() != 2 || !pair[0].isString() || !pair[1].isString())
            throw TokenizerError("a merge rule is not a pair of strings");

        return {pair[0].asString(), pair[1].asString()};
    }

    if (!entry.isString())
        throw TokenizerError("a merge rule is neither a string nor a pair");

    const auto rule = std::string_view {entry.asString()};
    const auto separator = rule.find(' ');

    if (separator == std::string_view::npos)
        throw TokenizerError("merge rule '" + std::string(rule) + "' has no space");

    return {rule.substr(0, separator), rule.substr(separator + 1)};
}

std::uint64_t mergeKey(TokenId left, TokenId right)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32)
           | static_cast<std::uint32_t>(right);
}

std::string readWholeFile(const std::filesystem::path& path)
{
    auto stream = std::ifstream {path, std::ios::binary};

    if (!stream)
        throw TokenizerError("cannot open '" + path.string() + "'");

    return std::string {std::istreambuf_iterator<char> {stream},
                        std::istreambuf_iterator<char> {}};
}
} // namespace

Tokenizer Tokenizer::fromFile(const std::filesystem::path& path)
{
    return fromJsonText(readWholeFile(path));
}

Tokenizer Tokenizer::fromJsonText(std::string_view jsonText)
{
    const auto root = Miro::Json::parse(jsonText);
    const auto& vocabValue = memberOrThrow(memberOrThrow(root, "model"), "vocab");
    const auto& mergesValue = memberOrThrow(memberOrThrow(root, "model"), "merges");

    if (!vocabValue.isObject())
        throw TokenizerError("model.vocab is not an object");

    if (!mergesValue.isArray())
        throw TokenizerError("model.merges is not an array");

    const auto* addedTokens = Miro::Json::find(root.asObject(), "added_tokens");
    const auto& vocab = vocabValue.asObject();

    auto highestId = TokenId {-1};

    for (const auto& [text, id]: vocab)
        highestId = std::max(highestId, asTokenId(id));

    if (addedTokens != nullptr && addedTokens->isArray())
    {
        for (const auto& entry: addedTokens->asArray())
            highestId = std::max(highestId, asTokenId(memberOrThrow(entry, "id")));
    }

    auto tokenizer = Tokenizer {};
    tokenizer.tokenTexts.resize(highestId + 1);
    tokenizer.specialFlags.resize(highestId + 1);

    for (const auto& [text, id]: vocab)
        tokenizer.tokenTexts[asTokenId(id)] = text;

    auto builder = SpecialTokensBuilder {};

    if (addedTokens != nullptr && addedTokens->isArray())
    {
        for (const auto& entry: addedTokens->asArray())
        {
            const auto id = asTokenId(memberOrThrow(entry, "id"));
            const auto& content = memberOrThrow(entry, "content").asString();

            tokenizer.tokenTexts[id] = content;
            tokenizer.specialFlags[id] = 1;
            builder.add(content, id);
        }
    }

    tokenizer.specialTokens = builder.build();

    tokenizer.tokensByText.reserve((std::size_t) tokenizer.tokenTexts.size());

    for (auto id = 0; id < tokenizer.tokenTexts.size(); ++id)
    {
        if (!tokenizer.tokenTexts[id].empty())
            tokenizer.tokensByText.emplace(tokenizer.tokenTexts[id], id);
    }

    const auto& mergeRules = mergesValue.asArray();
    tokenizer.merges.reserve((std::size_t) mergeRules.size());

    auto mergedText = std::string {};

    for (auto rank = 0; rank < mergeRules.size(); ++rank)
    {
        const auto [leftText, rightText] = splitMergeRule(mergeRules[rank]);

        mergedText.assign(leftText);
        mergedText += rightText;

        const auto left = tokenizer.tokenForText(leftText);
        const auto right = tokenizer.tokenForText(rightText);
        const auto merged = tokenizer.tokenForText(mergedText);

        if (left == invalidTokenId || right == invalidTokenId
            || merged == invalidTokenId)
            continue;

        tokenizer.merges.emplace(mergeKey(left, right), Merge {rank, merged});
    }

    return tokenizer;
}

int Tokenizer::size() const
{
    return tokenTexts.size();
}

bool Tokenizer::isSpecial(TokenId token) const
{
    return token >= 0 && token < specialFlags.size() && specialFlags[token] != 0;
}

std::string_view Tokenizer::textForToken(TokenId token) const
{
    if (token < 0 || token >= tokenTexts.size())
        return {};

    return tokenTexts[token];
}

TokenId Tokenizer::tokenForText(std::string_view tokenText) const
{
    const auto found = tokensByText.find(tokenText);
    return found == tokensByText.end() ? invalidTokenId : found->second;
}

// Every added token a Whisper tokenizer.json carries is spelled `<|...|>`, so
// that shape is what encode() recognises when one appears literally in the
// text it is asked to encode.
TokenId Tokenizer::specialTokenAt(std::string_view text, std::size_t position) const
{
    if (text.compare(position, specialTokenOpening.size(), specialTokenOpening) != 0)
        return invalidTokenId;

    const auto closing =
        text.find(specialTokenClosing, position + specialTokenOpening.size());

    if (closing == std::string_view::npos)
        return invalidTokenId;

    const auto end = closing + specialTokenClosing.size();
    const auto token = tokenForText(text.substr(position, end - position));

    return isSpecial(token) ? token : invalidTokenId;
}

void Tokenizer::encodePreToken(std::string_view preToken,
                               Vector<TokenId>& tokens) const
{
    struct Symbol
    {
        TokenId token = invalidTokenId;
        int previous = -1;
        int next = -1;
        bool alive = true;
    };

    struct Candidate
    {
        int rank = 0;
        int left = 0;
        TokenId leftToken = invalidTokenId;
        TokenId rightToken = invalidTokenId;

        bool operator>(const Candidate& other) const
        {
            return rank != other.rank ? rank > other.rank : left > other.left;
        }
    };

    const auto printable = ByteEncoding::toPrintable(preToken);
    auto symbols = Vector<Symbol> {};

    for (auto position = std::size_t {}; position < printable.size();)
    {
        const auto codePoint = Unicode::readCodePoint(printable, position);
        const auto length = static_cast<std::size_t>(codePoint.byteLength);
        const auto index = symbols.size();

        symbols.add(Symbol {
            tokenForText(std::string_view {printable}.substr(position, length)),
            index - 1,
            index + 1,
            true});

        position += length;
    }

    if (symbols.empty())
        return;

    symbols.back().next = -1;

    auto queue =
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> {};

    const auto pushCandidate = [&](int left)
    {
        if (left < 0 || symbols[left].next < 0)
            return;

        const auto right = symbols[left].next;
        const auto found =
            merges.find(mergeKey(symbols[left].token, symbols[right].token));

        if (found == merges.end())
            return;

        queue.push(Candidate {
            found->second.rank, left, symbols[left].token, symbols[right].token});
    };

    for (auto index = 0; index < symbols.size(); ++index)
        pushCandidate(index);

    while (!queue.empty())
    {
        const auto candidate = queue.top();
        queue.pop();

        auto& left = symbols[candidate.left];

        if (!left.alive || left.token != candidate.leftToken || left.next < 0)
            continue;

        const auto rightIndex = left.next;
        auto& right = symbols[rightIndex];

        if (right.token != candidate.rightToken)
            continue;

        const auto found = merges.find(mergeKey(left.token, right.token));

        if (found == merges.end())
            continue;

        left.token = found->second.merged;
        left.next = right.next;
        right.alive = false;

        if (left.next >= 0)
            symbols[left.next].previous = candidate.left;

        pushCandidate(candidate.left);
        pushCandidate(left.previous);
    }

    for (auto index = 0; index >= 0; index = symbols[index].next)
    {
        if (symbols[index].token != invalidTokenId)
            tokens.add(symbols[index].token);
    }
}

Vector<TokenId> Tokenizer::encode(std::string_view text) const
{
    auto tokens = Vector<TokenId> {};
    auto plainStart = std::size_t {};

    const auto encodePlainTextUpTo = [&](std::size_t end)
    {
        if (end <= plainStart)
            return;

        const auto plain = text.substr(plainStart, end - plainStart);

        for (auto preToken: PreTokenizer::split(plain))
            encodePreToken(preToken, tokens);
    };

    auto position = text.find(specialTokenOpening);

    while (position != std::string_view::npos)
    {
        const auto special = specialTokenAt(text, position);

        if (special == invalidTokenId)
        {
            position = text.find(specialTokenOpening, position + 1);
            continue;
        }

        encodePlainTextUpTo(position);
        tokens.add(special);
        position += textForToken(special).size();
        plainStart = position;
        position = text.find(specialTokenOpening, position);
    }

    encodePlainTextUpTo(text.size());
    return tokens;
}

std::string Tokenizer::decodeTokens(Span<const TokenId> tokens,
                                    bool keepSpecial) const
{
    auto text = std::string {};
    auto printable = std::string {};

    const auto flushPrintable = [&]
    {
        text += ByteEncoding::fromPrintable(printable);
        printable.clear();
    };

    for (auto token: tokens)
    {
        if (token < 0 || token >= tokenTexts.size())
            continue;

        if (isSpecial(token))
        {
            flushPrintable();

            if (keepSpecial)
                text += tokenTexts[token];

            continue;
        }

        printable += tokenTexts[token];
    }

    flushPrintable();
    return text;
}

std::string Tokenizer::decode(Span<const TokenId> tokens) const
{
    return decodeTokens(tokens, false);
}

std::string Tokenizer::decodeKeepingSpecialTokens(Span<const TokenId> tokens) const
{
    return decodeTokens(tokens, true);
}
} // namespace WSP
