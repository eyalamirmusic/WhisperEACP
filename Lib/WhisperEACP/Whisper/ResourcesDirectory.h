#pragma once

#include <filesystem>

namespace WSP
{
// Where the build's post-build copy puts what a binary ships
// (whisper_copy_resources in CMake/WhisperResources.cmake): Contents/Resources
// of a macOS bundle, and the directory the executable itself is in otherwise —
// a Windows build, or a macOS executable that is not a bundle. Resolved from
// the running process, so it holds wherever the build tree was moved to.
//
// Empty when the platform cannot say, which nothing here has seen happen; a
// caller that appends a name to it then looks for a relative path, and does
// not find one.
std::filesystem::path resourcesDirectory();
} // namespace WSP
