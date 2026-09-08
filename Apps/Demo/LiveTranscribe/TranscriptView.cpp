#include "TranscriptView.h"

#include <string_view>

namespace LiveTranscribe
{
namespace
{
constexpr auto textInset = 14.f;
constexpr auto paragraphGap = 8.f;
constexpr auto fallbackLineStep = 16.f;

constexpr auto whitespace = " \t\r\n";

WSP::Vector<std::string_view> wordsOf(std::string_view text)
{
    auto words = WSP::Vector<std::string_view> {};
    auto start = text.find_first_not_of(whitespace);

    while (start != std::string_view::npos)
    {
        const auto end = text.find_first_of(whitespace, start);
        words.add(text.substr(start, end - start));

        if (end == std::string_view::npos)
            break;

        start = text.find_first_not_of(whitespace, end);
    }

    return words;
}
} // namespace

void TranscriptView::setTranscript(const WSP::Vector<std::string>& committed,
                                   const std::string& pending)
{
    if (committed == paragraphs && pending == openParagraph)
        return;

    paragraphs = committed;
    openParagraph = pending;
    needsLayout = true;
}

void TranscriptView::layoutFor(float width)
{
    if (width <= 0.f || (width == layoutWidth && !needsLayout))
        return;

    layoutWidth = width;
    needsLayout = false;
    lines.clear();

    for (const auto& paragraph: paragraphs)
        appendWrapped(paragraph, false);

    appendWrapped(openParagraph, true);

    repaint();
}

// Greedy: words go onto the line until one does not fit, and a word wider than
// the whole line is left to overflow rather than broken, which a transcript
// never produces and a hyphenation rule would only get wrong.
void TranscriptView::appendWrapped(const std::string& paragraph, bool pending)
{
    const auto font = getHostFont();
    const auto available = layoutWidth - 2.f * textInset;

    auto opensParagraph = true;
    auto current = std::string {};

    const auto finishLine = [&]
    {
        lines.add(Line {current, pending, opensParagraph});
        opensParagraph = false;
    };

    for (auto word: wordsOf(paragraph))
    {
        if (current.empty())
        {
            current = word;
            continue;
        }

        auto candidate = current + ' ' + std::string {word};

        if (measureText(candidate, font) > available)
        {
            finishLine();
            current = word;

            continue;
        }

        current = std::move(candidate);
    }

    if (!current.empty())
        finishLine();
}

float TranscriptView::lineStep() const
{
    const auto* host = getHost();

    return host != nullptr ? host->getLineHeight(host->getFont()) : fallbackLineStep;
}

float TranscriptView::contentHeight() const
{
    const auto step = lineStep();

    auto height = 2.f * textInset;

    for (const auto& line: lines)
        height += step + (line.opensParagraph ? paragraphGap : 0.f);

    // The first paragraph opens with no gap above it, which is the one the loop
    // counted and the paint does not draw.
    return height - (lines.empty() ? 0.f : paragraphGap);
}

void TranscriptView::paint(UI::Graphics& g)
{
    const auto& theme = UI::defaultTheme();
    const auto step = lineStep();

    auto y = textInset;
    auto first = true;

    for (const auto& line: lines)
    {
        if (line.opensParagraph && !first)
            y += paragraphGap;

        first = false;

        g.setColour(line.pending ? theme.accent : theme.text);
        g.drawText(line.text,
                   UI::Rect {textInset, y, layoutWidth - 2.f * textInset, step});

        y += step;
    }
}
} // namespace LiveTranscribe
