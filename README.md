# omegahPetsc
omegah+petsc

## setup

```
module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load \
  cmake/3.15.4-mnqjvz6 \
  cuda/10.2

export OMPI_CXX=g++

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/gpfs/u/home/MPFS/MPFSsmth/barn-shared/cws/software/build-omegah-dcs-gcc74-cuda/install
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/gpfs/u/home/MPFS/MPFSsmth/barn-shared/cws/software/petsc/arch-aimos-gcc74Spectrum/lib/pkgconfig
```
