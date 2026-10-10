#include <WhisperEACP/Core/Core.h>
#include <eacp/Core/Platform/Platform.h>
#include <NanoTest/NanoTest.h>

using namespace nano;
using namespace WSP;

// Every executable of the tree is stamped with one company, so the folders eacp
// names after it, the Core ML cache above all, are one folder for all of them
// rather than one per binary. A test executable is stamped the way an app is.
auto tAppInfoNamesTheCompany = test("AppInfo/namesTheCompany") = []
{
    check(eacp::Platform::getCompanyName() == "eacp");
    check(eacp::Platform::getAppName() == "CoreTests");
    check(eacp::Platform::getAppVersion() == toString(getLibraryVersion()));
};
