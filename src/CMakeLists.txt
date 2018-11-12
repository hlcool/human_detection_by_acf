FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(OpenCV_DIR /usr/local/opencv341/share/OpenCV)
find_package(OpenCV 3.4 REQUIRED)
link_directories(${OpenCV_LIB_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

set(LIB_SRC ACFDetector.cpp
	    UBT_cup_detector.cpp
	    bbNms.cpp
	    box.cpp
	    CellArray.cpp
	    chnsCompute.cpp
	    chnsPyramid.cpp
	    convUtil.cpp
	    gradientUtil.cpp
	    imPad.cpp
	    imResample.cpp
	    rgbConvert.cpp
	    wrappers.cpp)
			
set(APP_SRC main.cpp)

add_library(cupDetector SHARED ${LIB_SRC})
target_link_libraries(cupDetector ${OpenCV_LIBS})

add_executable(demo ${APP_SRC})
target_link_libraries(demo ${OpenCV_LIBS} cupDetector)
add_dependencies(demo cupDetector)
