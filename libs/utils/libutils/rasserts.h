#pragma once

#include <stdexcept>

#include "string_utils.h"

int debugPoint(int line);

#define rassert(condition, id) if (!(condition)) { throw std::runtime_error("Assertion " + to_string(id) + " failed at line " + to_string(debugPoint(__LINE__)) + "!"); }
