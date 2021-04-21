cmake_minimum_required(VERSION 3.13.0)

project(OmegahPetsc VERSION 0.1 LANGUAGES CXX)

find_package(Omega_h REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc>3.7.0)

#hack to remove '-Werror' flag - petsc ex12 has many unused function variables
get_property(ohFlags TARGET Omega_h::omega_h PROPERTY INTERFACE_COMPILE_OPTIONS)
set(ohFlagsMod)
foreach(flags IN LISTS ohFlags)
  string(REPLACE "-Werror," "" flagsMod ${flags})
  list(APPEND ohFlagsMod ${flagsMod})
endforeach()
set_property(TARGET Omega_h::omega_h PROPERTY
  INTERFACE_COMPILE_OPTIONS "${ohFlagsMod}")

option(OMEGAH_PETSC_USE_CUDA OFF)
if(OMEGAH_PETSC_USE_CUDA AND NOT Omega_h_USE_CUDA)
  message(FATAL_ERROR "OMEGAH_PETSC_USE_CUDA cannot be enabled with a build of"
                      "Omega_h with CUDA disabled 'Omega_h_USE_CUDA=${Omega_h_USE_CUDA}'")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(ex12 ex12.cpp)
if(OMEGAH_PETSC_USE_CUDA)
  enable_language(CUDA)
  set_source_files_properties(ex12.cpp PROPERTIES LANGUAGE CUDA)
  set(HOST_FLAG_PREFIX "-Xcompiler")
endif()
target_link_libraries(ex12 Omega_h::omega_h PkgConfig::PETSC)

target_compile_options(ex12 PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
  ${HOST_FLAG_PREFIX} -Wno-unused-parameter>)

set(petscArgs
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

include(CTest)

function(mpiTest TESTNAME PROCS EXE)
  add_test(
    NAME ${TESTNAME}_p${PROCS}
    COMMAND mpirun -np ${PROCS} ${EXE} ${ARGN}
  )
endfunction(mpiTest)

mpiTest(24kxgc 2 ./ex12
    -mesh ${CMAKE_SOURCE_DIR}/24k.osh
    -vec_view vtk:24k_p2_sol.vtu:vtk_vtu
    -snes_view
    ${petscArgs})

mpiTest(24kxgc 4 ./ex12
    -mesh ${CMAKE_SOURCE_DIR}/24k.osh
    -vec_view vtk:24k_p4_sol.vtu:vtk_vtu
    -snes_view
    ${petscArgs})

mpiTest(box 2 ./ex12
  -mesh box
  -vec_view vtk:box_p2_sol.vtu:vtk_vtu
  -cells 3,3,0
  ${petscArgs})

mpiTest(box 4 ./ex12
  -mesh box
  -vec_view vtk:box_p4_sol.vtu:vtk_vtu
  -cells 4,4,0
  ${petscArgs})

mpiTest(box512 4 ./ex12
  -cells 512,512,0
  -mesh box
  -vec_view vtk:box_p4_sol.vtu:vtk_vtu
  -snes_view
  ${petscArgs})
  
mpiTest(picpart 4 ./ex12
  -mesh picpart 
  -picpart_path ${CMAKE_SOURCE_DIR}/xgc-picparts/24k/24k_full_bfs_4.ppm/24k_full_bfs_
  -vec_view vtk:24k_pic4_sol.vtu:vtk_vtu
  -snes_view
  -dm_view vtk:24k_pic4_dm.vtu:vtk_vtu
  -dm_plex_check_all
  ${petscArgs})

mpiTest(picpart 24 ./ex12
  -mesh picpart
  -picpart_path ${CMAKE_SOURCE_DIR}/xgc-picparts/24k/24k_bfs_bfs_24.ppm/24k_bfs_bfs_
  -vec_view vtk:24k_pic24_sol.vtu:vtk_vtu
  -snes_view
  -dm_view vtk:24k_pic24_dm.vtu:vtk_vtu
  -dm_plex_check_all
  ${petscArgs})
