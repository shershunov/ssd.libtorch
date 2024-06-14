add_executable (ysme "main.cpp" "main.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ysme PROPERTY CXX_STANDARD 20)
endif()