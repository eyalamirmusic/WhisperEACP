// The entry point every GPU-touching test executable that needs no model links
// instead of NanoTest's own: anything that reaches the device has to run inside
// eacp's event loop, which owns the run loop and the autorelease pool the Metal
// backend is written against.
//
// One copy, named by WHISPER_GPU_TEST_MAIN, rather than one per test module.
// ModelTestMain.cpp beside it is this plus the model fetch, for the modules that
// read the real weights.

#include "TestMain.h"

int main(int argc, char* argv[])
{
    return WSP::Testing::runTestsInEventLoop(argc, argv);
}
