#include "Version.h"

#include <eacp/Core/Utils/Strings.h>

namespace WSP
{
std::string toString(const Version& version)
{
    return eacp::Strings::concat(
        version.major, '.', version.minor, '.', version.patch);
}

Version getLibraryVersion()
{
    return {WHISPER_EACP_VERSION_MAJOR,
            WHISPER_EACP_VERSION_MINOR,
            WHISPER_EACP_VERSION_PATCH};
}
} // namespace WSP
