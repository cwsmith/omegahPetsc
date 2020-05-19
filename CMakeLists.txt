cmake_minimum_required(VERSION 3.13.0)

project(OmegahPetsc VERSION 0.1 LANGUAGES CXX CUDA)

find_package(Omega_h REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc>3.7.0)

#hack to remove '-Werror' flag - petsc ex12 has many unused function variables
get_property(ohFlags TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS)
list(GET ohFlags 0 ohFlags0)
string(REPLACE "-Werror," "" ohFlags0Mod ${ohFlags0})
list(GET ohFlags 1 ohFlags1)
list(APPEND ohFlags0Mod ${ohFlags1})
set_property(TARGET Omega_h::omega_h PROPERTY
  INTERFACE_COMPILE_OPTIONS "${ohFlags0Mod}")

add_executable(ex50 ex50.cpp)
set_source_files_properties(ex50.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(ex50 Omega_h::omega_h PkgConfig::PETSC)

add_executable(ex12 ex12.cpp)
set_source_files_properties(ex12.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(ex12 Omega_h::omega_h PkgConfig::PETSC)
