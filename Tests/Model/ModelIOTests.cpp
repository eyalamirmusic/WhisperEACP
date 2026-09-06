#include "Common.h"

#include <WhisperEACP/Model/ModelIO.h>

#include <limits>

using namespace nano;
using namespace WSP;
using namespace WSP::Testing;

namespace
{
std::int64_t integerFromJson(std::string_view text)
{
    return ModelIO::asInteger(Miro::Json::parse(text), "x");
}

// A Miro exception escaping counts as a failure, not as a throw: the whole
// point of the seam is that a caller catches exactly one type, and the message
// has to name the field so a bad config says which one.
template <typename Body>
bool throwsModelErrorNaming(std::string_view label, Body&& body)
{
    try
    {
        body();
    }
    catch (const ModelError& error)
    {
        return std::string_view {error.what()}.find(label) != std::string_view::npos;
    }
    catch (...)
    {
        return false;
    }

    return false;
}

Miro::Json::Object objectFromJson(std::string_view text)
{
    return Miro::Json::parse(text).asObject();
}
} // namespace

// data_offsets in a safetensors header are 64-bit byte offsets, so the reader
// has to carry every value int64 can hold, not only the ones a double names
// exactly.
auto tModelIOReadsPast2To53 = test("Model/IO/readsPast2To53") = []
{
    check(integerFromJson("9007199254740993") == 9007199254740993);
    check(integerFromJson("9223372036854775807")
          == std::numeric_limits<std::int64_t>::max());
    check(integerFromJson("-9223372036854775808")
          == std::numeric_limits<std::int64_t>::min());
};

auto tModelIOAcceptsIntegralDoubles = test("Model/IO/acceptsIntegralDoubles") = []
{
    check(integerFromJson("42.0") == 42);
    check(integerFromJson("-42.0") == -42);
    check(integerFromJson("0") == 0);
};

auto tModelIORejectsNonIntegers = test("Model/IO/rejectsNonIntegers") = []
{
    check(throwsModelErrorNaming("x", [] { return integerFromJson("1.5"); }));
    check(throwsModelErrorNaming("x", [] { return integerFromJson("\"1\""); }));
    check(throwsModelErrorNaming("x", [] { return integerFromJson("null"); }));
    check(throwsModelErrorNaming("x", [] { return integerFromJson("1e300"); }));
};

auto tModelIOIntegerFields = test("Model/IO/integerFields") = []
{
    const auto object = objectFromJson(R"({"n":7,"big":9007199254740993})");

    check(ModelIO::integerField(object, "n", "config") == 7);
    check(ModelIO::integerField(object, "big", "config") == 9007199254740993);
    check(ModelIO::intField(object, "n", "config") == 7);

    check(throwsModelErrorNaming(
        "missing", [&] { return ModelIO::integerField(object, "gone", "config"); }));

    check(throwsModelErrorNaming(
        "gone", [&] { return ModelIO::intField(object, "gone", "config"); }));

    check(throwsModelErrorNaming(
        "does not fit in an int",
        [&] { return ModelIO::intField(object, "big", "config"); }));

    check(ModelIO::intFieldOr(object, "gone", -1, "config") == -1);
    check(ModelIO::intFieldOr(object, "n", -1, "config") == 7);
};
