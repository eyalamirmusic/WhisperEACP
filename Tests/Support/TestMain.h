#pragma once

// nano::run inside eacp's event loop, shared by the entry points beside this
// file rather than copied into each: anything that reaches the GPU has to run in
// there, since the loop owns the run loop and the autorelease pool the Metal
// backend is written against.

#include <eacp/Core/App/App.h>

#include <NanoTest/NanoTest.h>

namespace WSP::Testing
{
// The exit code is carried back out of the loop through a static, which is what
// a capture-less callback needs; one process runs one suite, so there is nothing
// for it to collide with.
inline int runTestsInEventLoop(int argc, char* argv[])
{
    static auto argumentCount = 0;
    static char** argumentValues = nullptr;
    static auto exitCode = 0;

    argumentCount = argc;
    argumentValues = argv;

    eacp::Apps::run([] { exitCode = nano::run(argumentCount, argumentValues); });

    return exitCode;
}
} // namespace WSP::Testing
