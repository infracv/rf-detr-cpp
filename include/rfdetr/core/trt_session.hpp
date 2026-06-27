#pragma once
// TrtSession is an internal implementation detail — the canonical header lives
// in src/internal/.  This shim keeps apps/ and tools that historically included
// this path compiling without changes.
#include "../../../src/internal/trt_session.hpp"
