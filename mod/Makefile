MPI_ROOT?=/opt/ibm/spectrum_mpi

all: gpump_mod.mod

gpump_mod.mod: gpump_mod.F90
	$(MPI_ROOT)/bin/mpif90 -c gpump_mod.F90
	rm -f gpump_mod.o

clean:
	rm -f gpump_mod.mod *.o
