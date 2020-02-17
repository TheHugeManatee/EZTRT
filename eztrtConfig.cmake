# - Config file for the eztrt package
include(CMakeFindDependencyMacro)
find_dependency(OpenCV)
include("${CMAKE_CURRENT_LIST_DIR}/eztrtTargets.cmake")