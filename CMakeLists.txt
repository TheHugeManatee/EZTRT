#
# Main CMakelists for the boilerplate project.
#
# It aims to be a template and a CMake reference, and as such is documented as much as possible.
# While everything might not fit in any project, it should give good defaults and avoid CMake antipatterns.
#
# Project specific options :
#   - BUILD_USE_DOXYGEN
#   - BUILD_BUILD_TESTS (requires BUILD_TESTING set to ON)
# Other options might be available through the cmake scripts including (not exhaustive):
#   - BUILD_ENABLE_WARNINGS_SETTINGS
#   - BUILD_ENABLE_LTO
#
cmake_minimum_required(VERSION 3.12)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source.")
endif()

set(PROJECT_NAME "EZ TRT")
set(PROJECT_NAME_SHORT "eztrt") # short name without spaces, used as CMake namespace as well as file naming for the whole library collection
set(PROJECT_TARGETS "eztrtTargets")
set(PROJECT_VER_MAJOR 0)
set(PROJECT_VER_MINOR 1)
set(PROJECT_VER_PATCH 0)
set(${PROJECT_NAME_SHORT}_VERSION "${PROJECT_VER_MAJOR}.${PROJECT_VER_MINOR}.${PROJECT_VER_PATCH}")

project(${PROJECT_NAME} CXX)


#### Setup Installation directories
include(GNUInstallDirs)
set(CMAKE_INSTALL_PDBDIR ${CMAKE_INSTALL_BINDIR}) # typically, we want the PDB files in the same directory

# output all binaries and libs into a common folder across all libraries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "rwdbi")

############################
##  Modules and scripts   ##
############################

# Standard CMake modules
#include(CTest) # Must be called before adding tests but after calling project(). This automatically calls enable_testing() and configures ctest targets when using Make/Ninja
include(CMakeDependentOption)# This is a really useful scripts that creates options that depends on other options. It can even be used with generator expressions !

# Custom modules and scripts

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(LTO)
include(Warnings)
include(CopyDllsForDebug)

###############
##  OPTIONS  ##
###############

# You should try to give as much control over the project setup to the user.
# When modifying compile flags for example, if they are not mandatory, provide an option.

option(BUILD_USE_DOXYGEN "Add a doxygen target to generate the documentation" ON)

# Use your own option for tests, in case people use your library through add_subdirectory
cmake_dependent_option(BUILD_BUILD_TESTS
    "Enable Boilerplate project tests targets" ON # By default we want tests if CTest is enabled
    "BUILD_TESTING" OFF # Stay coherent with CTest variables
)

# External dependencies
add_subdirectory(external EXCLUDE_FROM_ALL)

# It is always easier to navigate in an IDE when projects are organized in folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# Check for LTO support (needs to be after project(...) )
find_lto(CXX)

add_subdirectory(eztrt-lib)
add_subdirectory(trt-host)

# Setup our project as the startup project for Visual so that people don't need to do it manually
set_directory_properties(PROPERTIES VS_STARTUP_PROJECT TRT-host)

#===========#
#   Tests   #
#===========#

# if(BP_BUILD_TESTS)
#     # Let the user add options to the test runner if needed
#     set(TEST_RUNNER_PARAMS "--force-colors=true" CACHE STRING "Options to add to our test runners commands")
#     # In a real project you most likely want to exclude test folders
#     # list(APPEND CUSTOM_COVERAGE_EXCLUDE "/test/")
#     add_subdirectory(tests)
#     # You can setup some custom variables and add them to the CTestCustom.cmake.in template to have custom ctest settings
#     # For example, you can exclude some directories from the coverage reports such as third-parties and tests
#     configure_file(
#         ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestCustom.cmake.in
#         ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake
#         @ONLY
#     )
# endif()

#############
## Doxygen ##
#############

if(BP_USE_DOXYGEN AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.9)
    find_package(Doxygen
        OPTIONAL_COMPONENTS dot mscgen dia
    )
    if(DOXYGEN_FOUND)
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
        doxygen_add_docs(
            doc
            README.md source
            COMMENT "Generate man pages"
        )
    endif()
endif()

####################
##  INSTALLATION  ##
####################
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME_SHORT}ConfigVersion.cmake"
  VERSION ${PROJECT_NAME_SHORT}_VERSION
  COMPATIBILITY SameMajorVersion)

install (FILES "${PROJECT_NAME_SHORT}Config.cmake" "${PROJECT_NAME_SHORT}ConfigVersion.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME_SHORT})

install(EXPORT ${PROJECT_TARGETS}
  FILE ${PROJECT_NAME_SHORT}Targets.cmake
  NAMESPACE ${PROJECT_NAME_SHORT}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME_SHORT})

export(PACKAGE ${PROJECT_NAME_SHORT})