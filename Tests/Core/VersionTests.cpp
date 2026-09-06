#include <WhisperEACP/Core/Core.h>
#include <NanoTest/NanoTest.h>

using namespace nano;
using namespace WSP;

auto tVersionOrdering = test("Version/ordering") = []
{
    check(Version {0, 1, 0} < Version {0, 2, 0});
    check(Version {0, 9, 9} < Version {1, 0, 0});
    check(Version {1, 2, 3} == Version {1, 2, 3});
};

auto tToString =
    test("Version/toString") = [] { check(toString(Version {1, 2, 3}) == "1.2.3"); };

// The version comes from project(), through compile definitions. A build that
// lost them would report 0.0.0 and nothing else would notice.
auto tLibraryVersionIsSet = test("Version/libraryVersionIsSet") = []
{ check(getLibraryVersion() > Version {}); };
