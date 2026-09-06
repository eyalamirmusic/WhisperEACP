#pragma once

#include <stdexcept>

namespace WSP
{
// Every way loading a model can fail, in one type: a truncated file, a header
// length that runs past the end, a byte range that leaves the blob, a shape
// whose element count disagrees with that range, a dtype we do not know, a
// missing config field.
//
// Exceptions rather than a result type, because that is what this tree already
// does — eacp::Files throws std::runtime_error and Miro::Json throws
// ParseError. Miro's is caught at the seam and rethrown as one of these, so a
// caller catches exactly one thing.
class ModelError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
} // namespace WSP
