cmake_minimum_required(VERSION 3.13.0)

project(OmegahPetsc VERSION 0.1 LANGUAGES CXX)

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

option(OMEGAH_PETSC_USE_CUDA OFF)

add_executable(ex12 ex12.cpp)
if(OMEGAH_PETSC_USE_CUDA)
  enable_language(CUDA)
  set_source_files_properties(ex12.cpp PROPERTIES LANGUAGE CUDA)
endif()
target_link_libraries(ex12 Omega_h::omega_h PkgConfig::PETSC)

target_compile_options(ex12 PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
  -Wno-unused-parameter>)

include(CTest)
add_test(ex12_24kxgc mpirun -np 2 ./ex12
  -mesh ${CMAKE_SOURCE_DIR}/24k.osh
  -run_type full
  -petscpartitioner_type parmetis
  -dm_refine 0
  -bc_type dirichlet
  -interpolate 1
  -petscspace_degree 1
  -ksp_type gmres
  -ksp_gmres_restart 100
  -ksp_rtol 1.0e-9
  -dm_mat_type is
  -pc_type bddc
  -snes_monitor_short
  -ksp_monitor_short
  -snes_converged_reason
  ::ascii_info_detail
  -ksp_converged_reason
  -snes_view
  -show_solution 0
  -log_view)

add_test(ex12_box mpirun -np 2 ./ex12
  -mesh box
  -run_type full
  -petscpartitioner_type parmetis
  -dm_refine 0
  -bc_type dirichlet
  -interpolate 1
  -petscspace_degree 1
  -ksp_type gmres
  -ksp_gmres_restart 100
  -ksp_rtol 1.0e-9
  -dm_mat_type is
  -pc_type bddc
  -snes_monitor_short
  -ksp_monitor_short
  -snes_converged_reason
  ::ascii_info_detail
  -ksp_converged_reason
  -show_solution 0
  -log_view)
