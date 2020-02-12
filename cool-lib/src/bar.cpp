#include "coolLib/bar.h"
#include "coolHeader/niftystuff.h"

#include <iostream>

namespace cool_lib
{

int bar(bool branch)
{
    if (branch)
    {
        std::cout << "This line will be untested, so that coverage is not "
                  << nifty::doANiftyThing(100) << "%\n";
    }
    else
    {
        std::cout << "This is the default behaviour and will be tested\n";
    }
    return 0;
}

} // namespace cool_lib
