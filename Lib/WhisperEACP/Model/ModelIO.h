#pragma once

#include <WhisperEACP/Core/Core.h>
#include <WhisperEACP/Model/ModelError.h>

#include <Miro/Json.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

// The reading every file this module loads has in common: get the bytes, get a
// JSON object out of them, and pull a field out of that object with a message
// naming what was being read when it was missing or the wrong shape. Miro's
// ParseError never escapes these — it comes back out as a ModelError.
namespace WSP::ModelIO
{
Vector<std::uint8_t> readFileBytes(const std::filesystem::path& path);

std::string_view textOf(Span<const std::uint8_t> bytes);

Miro::Json::Value parseObject(std::string_view text, std::string_view what);

const Miro::Json::Value& field(const Miro::Json::Object& object,
                               std::string_view key,
                               std::string_view what);

std::int64_t asInteger(const Miro::Json::Value& value, std::string_view what);
double asDouble(const Miro::Json::Value& value, std::string_view what);

std::int64_t integerField(const Miro::Json::Object& object,
                          std::string_view key,
                          std::string_view what);

int intField(const Miro::Json::Object& object,
             std::string_view key,
             std::string_view what);

int intFieldOr(const Miro::Json::Object& object,
               std::string_view key,
               int fallback,
               std::string_view what);

bool boolFieldOr(const Miro::Json::Object& object,
                 std::string_view key,
                 bool fallback,
                 std::string_view what);

std::string stringFieldOr(const Miro::Json::Object& object,
                          std::string_view key,
                          std::string_view fallback,
                          std::string_view what);

// Empty when the key is absent, so an optional list costs no branch at the call
// site. A present value that is not an array of integers is still an error.
Vector<int> intArrayFieldOr(const Miro::Json::Object& object,
                            std::string_view key,
                            std::string_view what);
} // namespace WSP::ModelIO
