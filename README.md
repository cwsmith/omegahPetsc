# omegahPetsc
omegah+petsc

The following instructions apply to the RPI AiMOS system.

## install petsc

We will be working in the `barn` directory.  If you are curious, you can read
more about the filesystem here:
https://secure.cci.rpi.edu/wiki/index.php?title=File_System

### clone the PETSc repo

```
cd ~/barn
git clone https://gitlab.com/petsc/petsc.git
```

The clone command will create a `petsc` directory and produce the following
output:

```
Cloning into 'petsc'...
remote: Enumerating objects: 878765, done.
remote: Counting objects: 100% (878765/878765), done.
remote: Compressing objects: 100% (202874/202874), done.
remote: Total 878765 (delta 673316), reused 876417 (delta 671285), pack-reused 0
Receiving objects: 100% (878765/878765), 211.03 MiB | 26.44 MiB/s, done.
Resolving deltas: 100% (673316/673316), done.
Checking out files: 100% (8528/8528), done.
```

### create an environment script

Enter the `petsc` directory:

```
cd petsc
```

and paste the following commands into a file named `envGnu74Cuda.sh`:

```
module use
/gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load \
  cmake/3.15.4-mnqjvz6 \
  cuda/10.2 \
  netlib-lapack \
  hdf5 \
  parmetis/4.0.3-int32-real32-nreyjmh \
  hypre/2.18.1-int32-rgc66ne

export OMPI_CXX=g++
export OMPI_CC=gcc
export OMPI_FC=gfortran
```

These commands setup the environment for building PETSc with the GNU compilers.

### create a configure script

Paste the following commands into a file named `arch-aimos.py`:

```
#!/usr/bin/python
if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=mpicc',
    '--with-cxx=mpiCC',
    '--with-fc=mpif90',
    '--with-shared-libraries=1',
    '--with-debugging=yes',
    '--COPTFLAGS=-g -O2 -mcpu=power9 -fPIC',
    '--CXXOPTFLAGS=-g -O2 -mcpu=power9 -fPIC',
    '--FOPTFLAGS=-g -O2 -mcpu=power9 -fPIC',
    '--with-blaslapack-lib=-L' + os.environ['NETLIB_LAPACK_ROOT'] + '/lib64
-lblas -llapack',
    '--with-cuda=1',
    '--with-cudac=nvcc',
    '--with-parmetis-dir=' + os.environ['PARMETIS_ROOT'],
    '--with-metis-dir=' + os.environ['METIS_ROOT'],
    '--with-make-np=16',
  ] 
  configure.petsc_configure(configure_options)
```

Make the script executable:

```
chmod +x arch-aimos.py
```

### configure the PETSc build

'Source' the environment script to export the settings within it to your active
terminal

```
source envGnu74Cuda.sh
```

Run the configure script:

```
./arch-aimos.py
```

This will take ~six minutes and will produce the following output if it
succeeded:

```
xxx=========================================================================xxx
 Configure stage complete. Now build PETSc libraries with:
   make PETSC_DIR=/gpfs/u/barn/MPFS/MPFSzhqg/petsc PETSC_ARCH=arch-aimos all
xxx=========================================================================xxx
```

### run make

As instructed by configure, we will now run make to compile the petsc libraries. 

```
make PETSC_DIR=/gpfs/u/barn/MPFS/MPFSzhqg/petsc PETSC_ARCH=arch-aimos all
```

This will take several minutes.  If it succeeds the following output will
appear:

```
Now to check if the libraries are working do:
make PETSC_DIR=/gpfs/u/barn/MPFS/MPFSzhqg/petsc PETSC_ARCH=arch-aimos check
=========================================
```

To run the 'make check' command PETSc provided we will need to allocate a
compute node.

## install omega_h

### download

```
cd ~/barn
git clone https://github.com/SNLComputation/omega_h.git
```

### setup

Create an environment script `~/barn/omega_h/envOmegaAimosGcc.sh` with the
following contents:

```
module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load \
  cmake/3.15.4-mnqjvz6 \
  cuda/10.2

export OMPI_CXX=g++
```


### build

```
source ~/barn/omega_h/envOmegaAimosGcc.sh 
mkdir ~/barn/build-omegah-AimosGcc74
cd !$
cmake ~/barn/omega_h \
-DCMAKE_INSTALL_PREFIX=$PWD/install \
-DBUILD_SHARED_LIBS=OFF \
-DOmega_h_USE_CUDA=on \
-DOmega_h_USE_MPI=on \
-DCMAKE_CUDA_HOST_COMPILER=`which mpicxx` \
-DCMAKE_CUDA_FLAGS="-arch=sm_70" 

make install -j16 
```

## install omegahPetsc

### download

```
cd ~/barn
git clone https://github.com/cwsmith/omegahPetsc.git
```

### setup

Create an environment file `~/barn/omegahPetsc/envOmegahPetscGccSpectrum.sh` with the
following contents:

```
module use /gpfs/u/software/dcs-spack-install/v0133gccSpectrum/lmod/linux-rhel7-ppc64le/gcc/7.4.0-1/
module load spectrum-mpi/10.3-doq6u5y
module load gcc/7.4.0/1
module load \
  cmake/3.15.4-mnqjvz6 \
  cuda/10.2

export OMPI_CXX=g++

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:~/barn/build-omegah-AimosGcc74/install/lib/cmake
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:~/barn/petsc/arch-aimos/lib/pkgconfig
```

## build

```
source ~/barn/omegahPetsc/envOmegahPetscGccSpectrum.sh
mkdir build-omegahPetsc
cd !$
cmake ../omegahPetsc/ -DCMAKE_CUDA_HOST_COMPILER=g++ -DCMAKE_CUDA_FLAGS="-arch=sm_70"
make
```
