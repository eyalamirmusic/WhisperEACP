#pragma once

#include <WhisperEACP/Core/Core.h>

#include <eacp/UI/UI.h>

#include <string>

namespace LiveTranscribe
{
namespace UI = eacp::UI;

// The transcript as text, wrapped by hand.
//
// eacp's painter draws a run on one line and measures one; there is no wrapped
// draw and no text area among the widgets, so the words are laid into lines
// here and paint() only draws what the layout produced. Which is also why the
// layout is driven from outside: the height a wrap produces is what the
// ScrollPanel holding this has to be told, and a component cannot resize
// itself.
//
// A closed segment is a paragraph in the theme's text colour; the open one
// follows it in the accent colour, because it is still being re-decoded and
// every word of it may still change.
class TranscriptView final : public UI::Component
{
public:
    void setTranscript(const WSP::Vector<std::string>& committed,
                       const std::string& pending);

    // Lays the text out at `width` if the width or the text has changed since
    // the last one. Cheap to call every frame; the wrap costs a measure per
    // word and only runs when something moved.
    void layoutFor(float width);

    float contentHeight() const;

    void paint(UI::Graphics& g) override;

private:
    struct Line
    {
        std::string text;
        bool pending = false;
        bool opensParagraph = false;
    };

    void appendWrapped(const std::string& paragraph, bool pending);

    // The host's, so the wrap measures against the face the paint will use.
    float lineStep() const;

    WSP::Vector<std::string> paragraphs;
    std::string openParagraph;

    WSP::Vector<Line> lines;
    float layoutWidth = 0.f;
    bool needsLayout = true;
};
} // namespace LiveTranscribe
