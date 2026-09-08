#include "MainPanel.h"

#include <MakeASound/MakeASound.h>

#include <array>
#include <cstdio>
#include <utility>

namespace LiveTranscribe
{
namespace
{
constexpr auto padding = 12.f;
constexpr auto rowHeight = 28.f;
constexpr auto meterHeight = 16.f;
constexpr auto lineHeight = 16.f;
constexpr auto smallText = 11.f;

std::string formatted(const char* format, double value)
{
    auto buffer = std::array<char, 48> {};
    std::snprintf(buffer.data(), buffer.size(), format, value);

    return buffer.data();
}

std::string milliseconds(double seconds)
{
    return formatted("%.0f ms", seconds * 1000.0);
}

std::string secondsText(double seconds)
{
    return formatted("%.2f s", seconds);
}

// Label::setText repaints whatever it is handed, and the status line is rebuilt
// thirty times a second out of numbers that change far less often.
void setLabelText(UI::Label& label, std::string text)
{
    if (label.getText() != text)
        label.setText(std::move(text));
}

void fillBox(UI::ComboBox& box,
             WSP::Vector<int>& ids,
             const MakeASound::UI::DropdownInfo& info)
{
    box.clear();
    ids.clear();

    auto selected = -1;

    for (const auto& item: info.items)
    {
        if (item.id == info.currentId)
            selected = ids.size();

        ids.add(item.id);
        box.addItem(item.label);
    }

    box.setSelectedIndex(selected);
}
} // namespace

MainRoot::MainRoot(Session& sessionToUse)
    : session(sessionToUse)
{
    title.setFontSize(15.f);
    status.setJustification(UI::Justification::Right);

    for (auto* label: {&status, &inputCaption, &channelCaption, &footer})
    {
        label->setColour(UI::defaultTheme().dimText);
        label->setFontSize(smallText);
    }

    errorLine.setColour({0.98f, 0.42f, 0.38f, 1.f});
    errorLine.setFontSize(smallText);

    startButton.setToggleable(true);
    transcriptScroll.setContent(transcript);

    wireControls();

    addChildren({title, status, inputCaption, deviceBox});
    addChildren({channelCaption, channelBox, startButton, clearButton});
    addChildren({meter, transcriptScroll, footer, errorLine});
}

void MainRoot::wireControls()
{
    startButton.onClick = [this] { toggleCapture(); };

    clearButton.onClick = [this]
    {
        session.clearTranscript();
        refreshTranscript();
    };

    deviceBox.onChange = [this](int index) { applyDeviceSelection(index); };
    channelBox.onChange = [this](int index) { applyChannelSelection(index); };
}

void MainRoot::refresh(bool rebuildDevices)
{
    if (rebuildDevices)
        rebuildDeviceBoxes();

    meter.setLevel(session.capture().level());

    applyStartButtonState();

    setLabelText(status, statusText());
    setLabelText(footer, footerText());
    setLabelText(errorLine, session.lastError());
}

void MainRoot::refreshTranscript()
{
    transcript.setTranscript(session.committed(), session.pending());
    layOutTranscript();
}

// An open list is the one thing a two-second rebuild must not walk over: the
// items behind the pointer would be replaced mid-click.
void MainRoot::rebuildDeviceBoxes()
{
    if (deviceBox.isPopupOpen() || channelBox.isPopupOpen())
        return;

    auto& audio = session.capture();
    const auto devices = MakeASound::UIDeviceManager {audio.deviceManager()};
    const auto currentDevice = audio.deviceId();

    fillBox(deviceBox, deviceIds, devices.makeInputDeviceDropdown(currentDevice));
    fillBox(channelBox,
            channelValues,
            devices.makeInputChannelDropdown(
                currentDevice, audio.firstChannel(), audio.channelCount()));
}

void MainRoot::applyStartButtonState()
{
    const auto available = canCapture();

    if (available != startIsAvailable)
    {
        startIsAvailable = available;
        startButton.setAccentColour(available ? UI::defaultTheme().accent
                                              : UI::defaultTheme().outline);
    }

    const auto capturing = session.isCapturing();

    if (capturing != startShowsStop)
    {
        startShowsStop = capturing;
        startButton.setText(capturing ? "Stop" : "Start");
    }

    startButton.setToggleState(capturing);
}

bool MainRoot::canCapture() const
{
    return session.isReady();
}

void MainRoot::toggleCapture()
{
    if (!canCapture())
    {
        startButton.setToggleState(false);
        return;
    }

    if (startButton.getToggleState())
    {
        if (!session.startCapture())
            startButton.setToggleState(false);
    }
    else
    {
        session.stopCapture();
        refreshTranscript();
    }

    refresh(false);
}

// Capture re-opens a running stream on either of these, so the only thing to do
// here is say which slice of which device the model should hear.
void MainRoot::applyDeviceSelection(int index)
{
    if (index >= 0 && index < deviceIds.size())
        session.capture().setDevice(deviceIds[index]);

    refresh(true);
}

void MainRoot::applyChannelSelection(int index)
{
    if (index >= 0 && index < channelValues.size())
    {
        const auto slice =
            MakeASound::UI::decodeChannelSelection(channelValues[index]);

        session.capture().setChannels(slice.firstChannel, slice.count);
    }

    refresh(true);
}

std::string MainRoot::statusText() const
{
    switch (session.state())
    {
        case ModelState::Loading:
            return "loading model...";

        case ModelState::NotBundled:
        case ModelState::Failed:
            return session.stateMessage();

        case ModelState::Ready:
            break;
    }

    const auto loaded = "model ready in " + secondsText(session.loadSeconds());
    const auto stream = session.capture().status();

    if (!stream.running)
        return loaded + "   idle";

    return deviceBox.getText() + "   " + std::to_string(stream.sampleRate) + " Hz   "
           + (stream.native ? "native 16 kHz" : "resampled") + "   block "
           + std::to_string(stream.blockSize) + "   encode "
           + milliseconds(session.lastEncodeSeconds()) + "   decode "
           + milliseconds(session.lastDecodeSeconds());
}

std::string MainRoot::footerText() const
{
    const auto stream = session.capture().status();
    const auto live = session.liveStats();

    return std::to_string(stream.blocks) + " blocks   "
           + std::to_string(stream.overflows) + " overflows   "
           + std::to_string(stream.dropped) + " dropped   "
           + std::to_string(live.runs) + " runs   "
           + secondsText(live.pendingSeconds) + " pending";
}

void MainRoot::paint(UI::Graphics& g)
{
    g.fillAll(UI::defaultTheme().background);
}

void MainRoot::resized()
{
    auto area = getLocalBounds().inset(padding);

    auto header = area.removeFromTop(22.f);
    title.setBounds(header.removeFromLeft(header.w * 0.3f));
    status.setBounds(header);

    area.removeFromTop(10.f);

    auto controls = area.removeFromTop(rowHeight);
    clearButton.setBounds(controls.removeFromRight(72.f));
    controls.removeFromRight(8.f);
    startButton.setBounds(controls.removeFromRight(80.f));
    controls.removeFromRight(14.f);

    inputCaption.setBounds(controls.removeFromLeft(40.f));
    deviceBox.setBounds(controls.removeFromLeft(controls.w * 0.55f));
    controls.removeFromLeft(12.f);
    channelCaption.setBounds(controls.removeFromLeft(56.f));
    channelBox.setBounds(controls);

    area.removeFromTop(10.f);
    meter.setBounds(area.removeFromTop(meterHeight));
    area.removeFromTop(10.f);

    errorLine.setBounds(area.removeFromBottom(lineHeight));
    footer.setBounds(area.removeFromBottom(lineHeight));
    area.removeFromBottom(8.f);

    transcriptScroll.setBounds(area);

    layOutTranscript();
}

// The wrap needs a width, and the panel holding it needs the height that wrap
// came to, so the two are set here rather than by either component alone.
void MainRoot::layOutTranscript()
{
    const auto width = transcriptScroll.getWidth();

    if (width <= 0.f)
        return;

    transcript.layoutFor(width);

    const auto height = transcript.contentHeight();

    transcript.setBounds(
        {0.f, -transcriptScroll.getScrollPosition(), width, height});

    // Clamped to the maximum, which is what keeps the newest words in view.
    transcriptScroll.setScrollPosition(height);
}

MainPanel::MainPanel(Session& session)
    : root(session)
{
    setFontPointSize(13.f);
    setBackgroundColour(UI::defaultTheme().background);
    setRootComponent(root);
}
} // namespace LiveTranscribe
