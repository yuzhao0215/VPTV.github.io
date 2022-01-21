# - Try to find Natural Point Camera SDK for windows
# once done this will define
# NP_CAMERASDK_FOUND - System has the Camera SDK
# NP_CAMERASDK_INCLUDE_DIRS - The Camera SDK include directories
# NP_CAMERASDK_LIBRARY_DIRS - The camera sdk library directories
# NP_CAMERASDK_LIBRARIES - The libraries needed to use the camera SDK
# NP_CAMERASDK_DEFINITIONS - Compiler switches

#find_path(NP_CAMERASKD_INCLUDE_DIR NAMES cameralibrary.h HINTS $ENV{NP_CAMERASDK}/include)
set(NP_CAMERASDK_INCLUDE_DIR $ENV{NP_CAMERASDK}/include)
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(STATUS "64 bit environment")
	find_library(NP_CAMERASDK_LIBRARY NAMES CameraLibrary2015x64D.lib HINTS $ENV{NP_CAMERASDK}/lib)
	add_definitions(-DWIN64)
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(STATUS "32 bit enironment")
	find_library(NP_CAMERASDK_LIBRARY NAMES cameralibrary HINTS $ENV{NP_CAMERASDK}/lib)
	add_definitions(-DWIN32)
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

set(NP_CAMERASDK_LIBRARIES ${NP_CAMERASDK_LIBRARY})
set(NP_CAMERASDK_INCLUDE_DIRS ${NP_CAMERASDK_INCLUDE_DIR})
set(NP_CAMERASDK_LIBRARY_DIRS $ENV{NP_CAMERASDK}/lib )
set(NP_CAMERASDK_LIBRARIES ${NP_CAMERASDK_LIBRARY_DIRS}/CameraLibrary2015x64D.lib)

add_definitions(-DCAMERALIBRARY_IMPORTS)
add_definitions(-DUSE_NP_CAMERASDK)
link_directories(${NP_CAMERASDK_LIBRARY_DIRS})
include_directories(${NP_CAMERASDK_INCLUDE_DIRS})

message(STATUS "NP CAMERA SDK ROOT: " $ENV{NP_CAMERASDK})
message(STATUS "INCLUDE: " ${NP_CAMERASDK_INCLUDE_DIRS})
message(STATUS "INCLUDE: " ${NP_CAMERASDK_LIBRARY_DIRS})
message(STATUS "LIBRARY: " ${NP_CAMERASDK_LIBRARIES})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NP_CAMERASDK DEFAULT_MSG
								  NP_CAMERASDK_LIBRARY NP_CAMERASDK_INCLUDE_DIR)
								  
mark_as_advanced(NP_CAMERASDK_INCLUDE_DIR NP_CAMERASDK_LIBRARY)