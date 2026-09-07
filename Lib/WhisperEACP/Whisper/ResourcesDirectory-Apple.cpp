#include "ResourcesDirectory.h"

#include <CoreFoundation/CoreFoundation.h>

#include <climits>
#include <string>

namespace WSP
{
namespace
{
// CoreFoundation owns the notion this file is after: for an executable inside
// a bundle the main bundle's resources directory is Contents/Resources, and
// for one that is not, CFBundleGetMainBundle still answers — with a bundle
// whose resources directory is the executable's own. That second answer is
// what makes a console tool and an .app resolve the same rule the build
// applied when it copied.
struct ReleasedOnExit
{
    CFURLRef url = nullptr;

    ~ReleasedOnExit()
    {
        if (url != nullptr)
            CFRelease(url);
    }
};
} // namespace

std::filesystem::path resourcesDirectory()
{
    const auto bundle = CFBundleGetMainBundle();

    if (bundle == nullptr)
        return {};

    const auto url = ReleasedOnExit {CFBundleCopyResourcesDirectoryURL(bundle)};

    if (url.url == nullptr)
        return {};

    auto path = std::string(PATH_MAX, '\0');

    if (!CFURLGetFileSystemRepresentation(
            url.url, true, reinterpret_cast<UInt8*>(path.data()), path.size()))
        return {};

    path.resize(path.find('\0'));

    return {path};
}
} // namespace WSP
