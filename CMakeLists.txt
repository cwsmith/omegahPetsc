cmake_minimum_required(VERSION 3.13.0)

project(OmegahPetsc VERSION 0.1 LANGUAGES CXX)

find_package(Omega_h REQUIRED)
enable_language(CUDA)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc>3.7.0)

add_executable(ex50 ex50.cpp)
target_link_libraries(ex50 Omega_h::omega_h PkgConfig::PETSC)
