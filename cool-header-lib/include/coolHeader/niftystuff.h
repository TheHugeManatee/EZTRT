#pragma once

#include <string>

namespace nifty {

    template <typename T>
    std::string doANiftyThing(const T& thing) {
        return std::to_string(thing);
    };
}