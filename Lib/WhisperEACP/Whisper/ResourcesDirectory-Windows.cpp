#include "ResourcesDirectory.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <string>

namespace WSP
{
namespace
{
// GetModuleFileNameW truncates to the buffer and reports its full length as
// the return value, so a path longer than the first guess is retried with room
// for it rather than returned cut off.
std::wstring executablePath()
{
    auto path = std::wstring(MAX_PATH, L'\0');

    for (;;)
    {
        const auto written = GetModuleFileNameW(
            nullptr, path.data(), static_cast<DWORD>(path.size()));

        if (written == 0)
            return {};

        if (written < path.size())
        {
            path.resize(written);
            return path;
        }

        path.resize(path.size() * 2);
    }
}
} // namespace

std::filesystem::path resourcesDirectory()
{
    const auto executable = executablePath();

    if (executable.empty())
        return {};

    return std::filesystem::path {executable}.parent_path();
}
} // namespace WSP
