#pragma once

#include <string>

namespace WSP
{
struct Version
{
    int major = 0;
    int minor = 0;
    int patch = 0;

    friend constexpr auto operator<=>(const Version&, const Version&) = default;
};

std::string toString(const Version& version);

Version getLibraryVersion();
} // namespace WSP
