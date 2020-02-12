/** @file main.cpp
 * Just a simple hello world using libfmt
 */
// The previous block is needed in every file for which you want to generate documentation

#include <iostream>

int main(int argc, char* argv[])
{
    if (argc) { std::cout << "The first argument was " << argv[0] << "\n"; }
    else
    {
        std::cout << "No arguments given.\n";
    }
    return 0;
}
