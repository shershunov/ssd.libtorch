cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ysme)

set(CMAKE_CXX_STANDARD 17)

set(Torch_DIR "C:/Program Files/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(OpenCV_DIR "C:/Program Files/opencv/build")
find_package(OpenCV REQUIRED)


add_executable(${PROJECT_NAME} "main.cpp" "train/train.h" "train/train.cpp" "model/model.h" "model/model.cpp" "utils/dataloader.h" "utils/dataloader.cpp" "utils/test.h" "utils/test.cpp" "train/loss.h" "train/loss.cpp"   "utils/utils.cpp")


if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif(MSVC)


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
file(COPY "C:/Program Files/opencv/build/x64/vc16/bin/opencv_world490d.dll" DESTINATION ${CMAKE_BINARY_DIR})