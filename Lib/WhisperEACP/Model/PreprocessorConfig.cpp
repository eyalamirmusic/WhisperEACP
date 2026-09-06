#include "PreprocessorConfig.h"

#include "ModelIO.h"

#include <eacp/GPU/Device/Device.h>

namespace WSP
{
namespace
{
constexpr auto preprocessorName = "preprocessor_config.json";

struct Matrix
{
    int rows = 0;
    int columns = 0;
    Vector<float> values;
};

Matrix parseMatrix(const Miro::Json::Object& object, std::string_view key)
{
    const auto& value = ModelIO::field(object, key, preprocessorName);

    if (!value.isArray())
        throw ModelError {"preprocessor_config.json mel_filters is not an array"};

    const auto& rows = value.asArray();
    auto matrix = Matrix {};
    matrix.rows = rows.size();

    for (const auto& row: rows)
    {
        if (!row.isArray())
            throw ModelError {"preprocessor_config.json mel_filters has a row "
                              "that is not an array"};

        const auto& columns = row.asArray();

        if (matrix.columns == 0)
            matrix.columns = columns.size();
        else if (columns.size() != matrix.columns)
            throw ModelError {"preprocessor_config.json mel_filters is ragged"};

        for (const auto& entry: columns)
            matrix.values.add(static_cast<float>(
                ModelIO::asDouble(entry, "preprocessor_config.json mel_filters")));
    }

    return matrix;
}

Vector<float> transposed(const Matrix& matrix)
{
    auto flipped = Vector<float> {};
    flipped.resize(matrix.values.size());

    for (auto row = 0; row < matrix.rows; ++row)
        for (auto column = 0; column < matrix.columns; ++column)
            flipped[column * matrix.rows + row] =
                matrix.values[row * matrix.columns + column];

    return flipped;
}

Vector<float> melMajorFilters(const Matrix& matrix, int melBins, int columns)
{
    if (matrix.rows == melBins && matrix.columns == columns)
        return matrix.values;

    if (matrix.rows == columns && matrix.columns == melBins)
        return transposed(matrix);

    throw ModelError {"preprocessor_config.json mel_filters is not "
                      + std::to_string(melBins) + " x " + std::to_string(columns)};
}
} // namespace

int PreprocessorConfig::melFilterColumns() const
{
    return fftSize / 2 + 1;
}

Span<const float> PreprocessorConfig::melFilterRow(int bin) const
{
    const auto columns = melFilterColumns();

    if (bin < 0 || bin >= melBins)
        return {};

    return {melFilters.data() + bin * columns, columns};
}

eacp::GPU::Buffer PreprocessorConfig::makeMelFilterBuffer() const
{
    return eacp::GPU::Device::shared().makeBuffer(
        melFilters.data(),
        static_cast<int>(melFilters.size() * sizeof(float)),
        eacp::GPU::BufferUsage::Storage);
}

PreprocessorConfig PreprocessorConfig::fromFile(const std::filesystem::path& path)
{
    const auto bytes = ModelIO::readFileBytes(path);
    return fromJson(ModelIO::textOf(bytes));
}

PreprocessorConfig PreprocessorConfig::fromJson(std::string_view text)
{
    const auto parsed = ModelIO::parseObject(text, preprocessorName);
    const auto& object = parsed.asObject();

    auto config = PreprocessorConfig {};

    config.sampleRate = ModelIO::intField(object, "sampling_rate", preprocessorName);
    config.fftSize = ModelIO::intField(object, "n_fft", preprocessorName);
    config.hopSize = ModelIO::intField(object, "hop_length", preprocessorName);
    config.melBins = ModelIO::intField(object, "feature_size", preprocessorName);
    config.chunkSeconds =
        ModelIO::intField(object, "chunk_length", preprocessorName);
    config.windowSamples = ModelIO::intField(object, "n_samples", preprocessorName);
    config.maxFrames = ModelIO::intField(object, "nb_max_frames", preprocessorName);

    if (config.sampleRate <= 0 || config.hopSize <= 0 || config.melBins <= 0)
        throw ModelError {"preprocessor_config.json has a non-positive rate, hop "
                          "or mel bin count"};

    if (config.fftSize <= 0 || config.fftSize % 2 != 0)
        throw ModelError {"preprocessor_config.json n_fft is not a positive even "
                          "number"};

    config.melFilters = melMajorFilters(parseMatrix(object, "mel_filters"),
                                        config.melBins,
                                        config.melFilterColumns());

    return config;
}
} // namespace WSP
