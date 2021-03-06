set(TARGET_NAME eztrt)
set(TARGET_HEADER_SUBDIR "eztrt/")

add_library(${TARGET_NAME} 
  src/base.cpp
  src/model.cpp
  src/util.cpp
)

find_package(TensorRT REQUIRED COMPONENTS nvparsers nvonnxparser)

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

find_package(OpenCV REQUIRED COMPONENTS core videoio imgproc)

target_link_libraries(${TARGET_NAME} 
PUBLIC
  TensorRT::TensorRT ${CUDA_LIBRARIES}
  opencv_core
  opencv_videoio
  opencv_imgproc
PRIVATE
  ext_libs
)

# Define a macro that allows consumers of this lib to know they link against it
target_compile_definitions(${TARGET_NAME} PUBLIC HAS_EZTRT)

# generate the export header for this library
include(GenerateExportHeader)
generate_export_header(${TARGET_NAME} EXPORT_FILE_NAME "${TARGET_HEADER_SUBDIR}${TARGET_NAME}_export.h")

# CMake scripts extensions
target_set_warnings(${TARGET_NAME} ENABLE ALL AS_ERROR ALL DISABLE Annoying C4100)
target_enable_lto(${TARGET_NAME} optimized) #enable lto if available for non-debug configurations
target_compile_features(${TARGET_NAME} PUBLIC cxx_std_17)
set_target_properties(${TARGET_NAME} PROPERTIES CXX_STANDARD_REQUIRED on)

# group this lib into the 'libs' group for some IDEs
set_target_properties (${TARGET_NAME} PROPERTIES FOLDER libs)

# add headers as sources automatically - this makes them show up in some IDEs
# but is not strictly necessary (unless you are using CMAKE_AUTOMOC)
file(GLOB_RECURSE HEADERS "include/*.h")
target_sources(${TARGET_NAME} PRIVATE ${HEADERS})

# Set the include directories of this library
target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${CUDA_INCLUDE_DIRS}
)


###### INSTALLATION

# Install in the desired folder
add_library(eztrt::eztrt ALIAS eztrt)
install(TARGETS ${TARGET_NAME}
  EXPORT ${PROJECT_TARGETS}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install includes
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# Install PDB file (MSVC debug symbols) for consumers of this library, if this library is 
# a shared library
get_target_property(target_type ${TARGET_NAME} TYPE)
if(target_type STREQUAL SHARED_LIBRARY)
  install(FILES $<TARGET_PDB_FILE:${TARGET_NAME}> DESTINATION ${CMAKE_INSTALL_PDBDIR} OPTIONAL)
endif()
# Install the export header
install(
  FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_HEADER_SUBDIR}${TARGET_NAME}_export.h 
  DESTINATION 
    ${CMAKE_INSTALL_INCLUDEDIR}/${TARGET_HEADER_SUBDIR}
)