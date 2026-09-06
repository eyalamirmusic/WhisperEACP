#include "SpecialTokens.h"

#include <charconv>
#include <cmath>

namespace WSP
{
namespace
{
constexpr auto specialPrefix = std::string_view {"<|"};
constexpr auto specialSuffix = std::string_view {"|>"};

std::string_view innerNameOf(std::string_view content)
{
    if (!content.starts_with(specialPrefix) || !content.ends_with(specialSuffix)
        || content.size() < specialPrefix.size() + specialSuffix.size() + 1)
        return {};

    return content.substr(specialPrefix.size(),
                          content.size() - specialPrefix.size()
                              - specialSuffix.size());
}

bool isLanguageCode(std::string_view name)
{
    if (name.size() < 2 || name.size() > 3)
        return false;

    for (auto character: name)
    {
        if (character < 'a' || character > 'z')
            return false;
    }

    return true;
}

bool readSeconds(std::string_view name, float& seconds)
{
    const auto dot = name.find('.');

    if (dot == std::string_view::npos || dot == 0 || dot + 1 == name.size())
        return false;

    for (auto character: name)
    {
        if (character != '.' && (character < '0' || character > '9'))
            return false;
    }

    auto whole = 0;
    auto fraction = 0;
    std::from_chars(name.data(), name.data() + dot, whole);
    std::from_chars(name.data() + dot + 1, name.data() + name.size(), fraction);

    const auto fractionDigits = name.size() - dot - 1;
    auto scale = 1.0f;

    for (auto i = std::size_t {}; i < fractionDigits; ++i)
        scale *= 10.0f;

    seconds = static_cast<float>(whole) + static_cast<float>(fraction) / scale;
    return true;
}
} // namespace

bool SpecialTokens::isTimestamp(TokenId token) const
{
    return firstTimestamp != invalidTokenId && token >= firstTimestamp
           && token < firstTimestamp + timestampCount;
}

TokenId SpecialTokens::timestampForSeconds(float seconds) const
{
    if (timestampCount == 0 || timestampInterval <= 0.0f)
        return invalidTokenId;

    const auto step = static_cast<int>(std::lround(seconds / timestampInterval));

    if (step < 0 || step >= timestampCount)
        return invalidTokenId;

    return firstTimestamp + step;
}

float SpecialTokens::secondsForTimestamp(TokenId token) const
{
    if (!isTimestamp(token))
        return -1.0f;

    return static_cast<float>(token - firstTimestamp) * timestampInterval;
}

TokenId SpecialTokens::forLanguage(std::string_view code) const
{
    const auto found = languages.find(code);
    return found == languages.end() ? invalidTokenId : found->second;
}

std::string_view SpecialTokens::languageOf(TokenId token) const
{
    for (const auto& [code, id]: languages)
    {
        if (id == token)
            return code;
    }

    return {};
}

void SpecialTokensBuilder::add(std::string_view content, TokenId token)
{
    const auto name = innerNameOf(content);

    if (name.empty())
        return;

    auto seconds = 0.0f;

    if (readSeconds(name, seconds))
    {
        if (tokens.firstTimestamp == invalidTokenId || token < tokens.firstTimestamp)
        {
            tokens.firstTimestamp = token;
            firstTimestampSeconds = seconds;
        }

        if (token > lastTimestamp)
        {
            lastTimestamp = token;
            lastTimestampSeconds = seconds;
        }

        ++tokens.timestampCount;
        return;
    }

    if (name == "endoftext")
        tokens.endOfText = token;
    else if (name == "startoftranscript")
        tokens.startOfTranscript = token;
    else if (name == "startofprev")
        tokens.startOfPrevious = token;
    else if (name == "startoflm")
        tokens.startOfLanguageModel = token;
    else if (name == "translate")
        tokens.translate = token;
    else if (name == "transcribe")
        tokens.transcribe = token;
    else if (name == "notimestamps")
        tokens.noTimestamps = token;
    else if (name == "nospeech" || name == "nocaptions")
        tokens.noSpeech = token;
    else if (isLanguageCode(name))
        tokens.languages.emplace(std::string(name), token);
}

SpecialTokens SpecialTokensBuilder::build() const
{
    auto built = tokens;

    if (built.timestampCount > 1)
    {
        const auto span = lastTimestampSeconds - firstTimestampSeconds;
        const auto steps = static_cast<float>(lastTimestamp - built.firstTimestamp);
        built.timestampInterval = span / steps;
    }

    return built;
}
} // namespace WSP
