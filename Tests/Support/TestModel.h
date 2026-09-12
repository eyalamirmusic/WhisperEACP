#pragma once

// The model a test executable needs, fetched before the suite runs.
//
// The entry point in ModelTestMain.cpp is what calls this, and that is the whole
// of the ordering rule: OnlineResource::fetch pumps the event loop, which a
// console main may do and an event-loop callback may not, so the fetch happens
// before eacp::Apps::run rather than inside a test. A test that wants to say
// something about the fetch reads what it left here instead.

#include <WhisperEACP/Model/ModelFetch.h>

#include <cstdio>

namespace WSP::Testing
{
// What the fetch at the top of this process found and did, for the tests in
// Tests/Model that are about the fetch rather than about the files.
struct ModelFetchRecord
{
    // Whether the four were already on disk when this process started, which is
    // what makes "and it downloaded nothing" an assertion rather than a wish.
    bool availableBefore = false;

    bool ran = false;
    ModelFetch::Outcome outcome;
};

// One per process, whichever translation unit asks: the static of an inline
// function is shared across them all.
inline ModelFetchRecord& testModelFetch()
{
    static auto record = ModelFetchRecord {};

    return record;
}

// Fetched once per process into the one directory every binary in this tree
// shares, so the first executable of a suite run downloads 151 MB and the rest
// find it. Freshness::trust because the URLs name a commit: the bytes at one
// cannot change, so a copy on disk is current by construction and a run with the
// model already there asks the network nothing.
//
// A failure is printed and swallowed. A machine with no network is the case the
// skips were always written for, and hasWhisperModel() is what each test asks; a
// test binary that failed on a missing download would be a test of the network.
inline void fetchTestModel()
{
    auto& record = testModelFetch();

    record.availableBefore = ModelFetch::isAvailable();
    record.ran = true;

    if (!record.availableBefore)
    {
        std::printf("fetching %s into %s\n",
                    ModelFetch::repository,
                    ModelFetch::directory().string().c_str());
        std::fflush(stdout);
    }

    record.outcome = ModelFetch::fetch(
        ModelFetch::Freshness::trust,
        [](const ModelFetch::Progress& progress)
        {
            using Stage = eacp::OnlineResource::Progress::Stage;

            if (progress.file.stage == Stage::done)
                std::printf("  %s\n", ModelFetch::progressText(progress).c_str());
        });

    if (!record.outcome.ok)
        std::printf("  no model, so the tests that need one will skip: %s\n",
                    record.outcome.error.c_str());

    std::fflush(stdout);
}
} // namespace WSP::Testing
