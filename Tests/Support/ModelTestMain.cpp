// The entry point for a test executable that reads the real tiny.en weights:
// the model fetched first, then the suite inside eacp's event loop.
//
// The order is the point, and it is why this is a second entry point rather than
// a flag on the first. OnlineResource::fetch pumps the event loop until the
// transfer is done, which is exactly what a console main may do and an
// event-loop callback may not, so the fetch has to happen before
// eacp::Apps::run — and a test body runs inside it.
//
// Named by WHISPER_MODEL_TEST_MAIN, and the modules that list it do not also
// list WHISPER_GPU_TEST_MAIN: this is that one with the fetch in front.

#include "TestMain.h"
#include "TestModel.h"

#include <string_view>

namespace
{
// NanoTest's CMake discovers a binary's test cases by running it with
// --list-tests after every link, and takes the whole of its stdout as the list
// of names. So a run that is only being asked what tests it has must fetch
// nothing and print nothing: otherwise `cmake --build` downloads 151 MB — the
// very thing this change took out of the build — and files four progress lines
// as test cases.
bool isListingTests(int argc, char* argv[])
{
    for (auto index = 1; index < argc; ++index)
        if (std::string_view {argv[index]} == "--list-tests")
            return true;

    return false;
}
} // namespace

int main(int argc, char* argv[])
{
    if (!isListingTests(argc, argv))
        WSP::Testing::fetchTestModel();

    return WSP::Testing::runTestsInEventLoop(argc, argv);
}
