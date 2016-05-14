CXX = icpc
CXXFLAGS = -qopenmp -mkl -qopt-report=4

OBJECTS = main.o
MICOBJECTS = main.oMIC

LD = icpc
LDFLAGS = -qopenmp -mkl

.SUFFIXES: .o .oMIC .cc

all:
	$(CXX) -xhost $(CXXFLAGS) -DLU_N=32 -o runme-32 main.cc
	$(CXX) -xhost $(CXXFLAGS) -DLU_N=64 -o runme-64 main.cc
	$(CXX) -xhost $(CXXFLAGS) -DLU_N=96 -o runme-96 main.cc
	$(CXX) -xhost $(CXXFLAGS) -DLU_N=128 -o runme-128 main.cc
	$(CXX) -xhost $(CXXFLAGS) -DLU_N=160 -o runme-160 main.cc
	$(CXX) -xhost $(CXXFLAGS) -DLU_N=192 -o runme-192 main.cc
	$(CXX) -xhost $(CXXFLAGS) -DLU_N=224 -o runme-224 main.cc
	$(CXX) -xhost $(CXXFLAGS) -DLU_N=256 -o runme-256 main.cc
	$(CXX) -mmic  $(CXXFLAGS) -DLU_N=32 -o runmeMIC-32 main.cc
	$(CXX) -mmic  $(CXXFLAGS) -DLU_N=64 -o runmeMIC-64 main.cc
	$(CXX) -mmic  $(CXXFLAGS) -DLU_N=96 -o runmeMIC-96 main.cc
	$(CXX) -mmic  $(CXXFLAGS) -DLU_N=128 -o runmeMIC-128 main.cc
	$(CXX) -mmic  $(CXXFLAGS) -DLU_N=160 -o runmeMIC-160 main.cc
	$(CXX) -mmic  $(CXXFLAGS) -DLU_N=192 -o runmeMIC-192 main.cc
	$(CXX) -mmic  $(CXXFLAGS) -DLU_N=224 -o runmeMIC-224 main.cc
	$(CXX) -mmic  $(CXXFLAGS) -DLU_N=256 -o runmeMIC-256 main.cc

clean: 
	rm -f main.o* runme* *.optrpt

run-cpu: 
	KMP_AFFINITY=scatter MKL_NUM_THREADS=1 ./runme-32
	KMP_AFFINITY=scatter MKL_NUM_THREADS=1 ./runme-64
	KMP_AFFINITY=scatter MKL_NUM_THREADS=1 ./runme-96
	KMP_AFFINITY=scatter MKL_NUM_THREADS=1 ./runme-128
	KMP_AFFINITY=scatter MKL_NUM_THREADS=1 ./runme-160
	KMP_AFFINITY=scatter MKL_NUM_THREADS=1 ./runme-192
	KMP_AFFINITY=scatter MKL_NUM_THREADS=1 ./runme-224
	KMP_AFFINITY=scatter MKL_NUM_THREADS=1 ./runme-256

run-mic: 
	ssh mic0 "MKL_NUM_THREADS=1 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC-32"
	ssh mic0 "MKL_NUM_THREADS=1 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC-64"
	ssh mic0 "MKL_NUM_THREADS=1 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC-96"
	ssh mic0 "MKL_NUM_THREADS=1 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC-128"
	ssh mic0 "MKL_NUM_THREADS=1 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC-160"
	ssh mic0 "MKL_NUM_THREADS=1 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC-192"
	ssh mic0 "MKL_NUM_THREADS=1 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC-224"
	ssh mic0 "MKL_NUM_THREADS=1 LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC-256"
