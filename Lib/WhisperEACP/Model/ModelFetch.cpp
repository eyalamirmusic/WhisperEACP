#include "ModelFetch.h"

#include <eacp/Core/Threads/EventLoop.h>
#include <eacp/Core/Utils/StdPath.h>

#include <cstdio>
#include <exception>
#include <utility>

namespace WSP
{
using eacp::FilePath;
using eacp::OnlineResource;

namespace
{
// The company and app the shared directory is named after, rather than the
// running executable's own names — see ModelFetch::directory().
constexpr auto company = "eacp";
constexpr auto application = "WhisperEACP";
constexpr auto modelFolder = "tiny.en";

constexpr auto host = "https://huggingface.co";

OnlineResource::Options optionsFor(const OnlineResource::Info& info,
                                   ModelFetch::Freshness freshness)
{
    auto options = OnlineResource::Options {info};

    options.directory = ModelFetch::directory();
    options.freshness = freshness;

    return options;
}

OnlineResource::Info modelResource(const char* fileName)
{
    auto info = OnlineResource::Info {};

    info.name = std::string {ModelFetch::repository} + " " + fileName;
    info.url = std::string {host} + "/" + ModelFetch::repository + "/resolve/"
               + ModelFetch::revision + "/" + fileName;
    info.fileName = fileName;
    info.version = ModelFetch::revision;

    return info;
}

std::string megabytes(std::int64_t bytes)
{
    char text[32] = {};
    std::snprintf(text, sizeof(text), "%.1f", (double) bytes / 1.0e6);

    return text;
}

std::string percent(float fraction)
{
    char text[16] = {};
    std::snprintf(text, sizeof(text), "%.0f%%", fraction * 100.0f);

    return text;
}
} // namespace

std::filesystem::path ModelFetch::directory()
{
    return eacp::toStdPath(FilePath::appSupportDirectory(company, application)
                           / "Models" / modelFolder);
}

Vector<OnlineResource::Info> ModelFetch::resources()
{
    auto infos = Vector<OnlineResource::Info> {};

    infos.add(modelResource("config.json"));
    infos.add(modelResource("preprocessor_config.json"));
    infos.add(modelResource("tokenizer.json"));

    // Last, and it is 98% of the bytes: a run with no network fails on a 2 kB
    // transfer rather than part-way through 151 MB.
    infos.add(modelResource("model.safetensors"));

    return infos;
}

bool ModelFetch::isAvailable()
{
    for (const auto& info: resources())
        if (!OnlineResource {info, directory()}.isAvailable())
            return false;

    return true;
}

std::string ModelFetch::progressText(const Progress& progress)
{
    if (progress.fileIndex < 1)
        return "looking for the model";

    const auto place = progress.fileName + " (" + std::to_string(progress.fileIndex)
                       + " of " + std::to_string(resourceCount) + ")";

    if (progress.file.stage == OnlineResource::Progress::Stage::done)
        return place + "   done";

    if (progress.file.fraction < 0.0f)
        return place + "   " + megabytes(progress.file.bytesReceived) + " MB so far";

    return place + "   " + megabytes(progress.file.bytesReceived) + " of "
           + megabytes(progress.file.totalBytes) + " MB   "
           + percent(progress.file.fraction);
}

ModelFetch::Outcome ModelFetch::fetch(Freshness freshness,
                                      ProgressCallback onProgress)
{
    auto outcome = Outcome {};
    outcome.directory = directory();

    const auto all = resources();

    for (auto index = 0; index < all.size(); ++index)
    {
        const auto& info = all[index];
        auto options = optionsFor(info, freshness);

        options.onProgress = [&, index](const OnlineResource::Progress& file)
        { onProgress({index + 1, info.fileName, file}); };

        try
        {
            outcome.downloaded |= OnlineResource::fetch(options).downloaded;
        }
        catch (const std::exception& failure)
        {
            outcome.error = failure.what();
            return outcome;
        }
    }

    outcome.ok = true;

    return outcome;
}

void ModelFetch::Download::start(Freshness freshnessToUse)
{
    if (running)
        return;

    freshness = freshnessToUse;
    queue = resources();
    index = 0;
    running = true;

    outcome = Outcome {};
    outcome.directory = directory();

    fetchNext();
}

void ModelFetch::Download::cancel()
{
    if (current != nullptr)
        current->cancel();
}

ModelFetch::Progress ModelFetch::Download::progress() const
{
    auto result = Progress {};

    if (current == nullptr)
        return result;

    result.fileIndex = index;
    result.fileName = current->info().fileName;
    result.file = current->progress();

    return result;
}

void ModelFetch::Download::fetchNext()
{
    if (index >= queue.size())
    {
        outcome.ok = true;
        finish();

        return;
    }

    const auto& info = queue[index];
    ++index;

    current = std::make_unique<OnlineResource>(info, directory(), freshness);

    current->start().then([this](const OnlineResource::Result& result)
                          { fileFinished(result); });
}

void ModelFetch::Download::fileFinished(const OnlineResource::Result& result)
{
    outcome.downloaded |= result.downloaded;
    outcome.cancelled = result.cancelled;

    if (result.ok)
    {
        postNextStep();

        return;
    }

    outcome.error = result.cancelled ? "cancelled" : result.error;
    finish();
}

void ModelFetch::Download::postNextStep()
{
    auto token = std::weak_ptr<int> {lifetime};

    eacp::Threads::callAsync(
        [this, token]
        {
            if (!token.expired())
                fetchNext();
        });
}

void ModelFetch::Download::finish()
{
    current.reset();
    running = false;

    onFinished(outcome);
}
} // namespace WSP
