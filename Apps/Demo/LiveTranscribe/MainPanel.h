#pragma once

#include "LevelMeter.h"
#include "Session.h"
#include "TranscriptView.h"

#include <eacp/UI/UI.h>

#include <string>

namespace LiveTranscribe
{
namespace UI = eacp::UI;

// The window's one component tree: what to listen to, whether to listen, how
// loud it is, and what it said.
class MainRoot final : public UI::Component
{
public:
    explicit MainRoot(Session& sessionToUse);

    // Everything that changes on a tick. `rebuildDevices` re-enumerates the
    // machine's inputs, which is why the timer only asks for it every few
    // seconds and whenever the device sent a notification.
    void refresh(bool rebuildDevices);

    // The transcript, which changes far less often than the meter does and
    // costs a wrap when it changes.
    void refreshTranscript();

    void paint(UI::Graphics& g) override;
    void resized() override;

private:
    void wireControls();
    void rebuildDeviceBoxes();
    void layOutTranscript();

    void toggleCapture();
    void applyDeviceSelection(int index);
    void applyChannelSelection(int index);

    std::string statusText() const;
    std::string footerText() const;

    // Whether Start can do anything: no model means nothing to transcribe, and
    // the widget tier has no disabled state, so the button says so by going the
    // colour of an outline and swallowing the click.
    bool canCapture() const;
    void applyStartButtonState();

    Session& session;

    UI::Label title {"Live Transcribe"};
    UI::Label status;

    UI::Label inputCaption {"input"};
    UI::Label channelCaption {"channels"};

    UI::ComboBox deviceBox {"device"};
    UI::ComboBox channelBox {"channels"};

    UI::Button startButton {"Start"};
    UI::Button clearButton {"Clear"};

    LevelMeter meter;

    UI::ScrollPanel transcriptScroll;
    TranscriptView transcript;

    UI::Label footer;
    UI::Label errorLine;

    // Parallel to the boxes' rows, since a dropdown item carries an id and a
    // ComboBox reports an index.
    WSP::Vector<int> deviceIds;
    WSP::Vector<int> channelValues;

    // What the button is already showing, so a tick that changed nothing about
    // it does not repaint it.
    bool startIsAvailable = false;
    bool startShowsStop = false;
};

class MainPanel final : public UI::ComponentHost
{
public:
    explicit MainPanel(Session& session);

    void refresh(bool rebuildDevices) { root.refresh(rebuildDevices); }
    void refreshTranscript() { root.refreshTranscript(); }

private:
    MainRoot root;
};
} // namespace LiveTranscribe
