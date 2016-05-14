CXX = icpc
CXXFLAGS = -qopenmp -mkl -qopt-report=4

OBJECTS = main.o
MICOBJECTS = main.oMIC

LD = icpc
LDFLAGS = -qopenmp -mkl

.SUFFIXES: .o .oMIC .cc

.cc.o:
	$(CXX) -c -xhost $(CXXFLAGS) -o "$@" "$<"

.cc.oMIC:
	$(CXX) -c -mmic $(CXXFLAGS) -o "$@" "$<"

all: runme runmeMIC

runme: $(OBJECTS)
	$(LD) $(LDFLAGS) -o runme $(OBJECTS)

runmeMIC: $(MICOBJECTS)
	$(LD) $(LDFLAGS) -mmic -o runmeMIC $(MICOBJECTS)

clean: 
	rm -f $(OBJECTS) $(MICOBJECTS) runme runmeMIC *.optrpt

run-cpu: runme
	KMP_AFFINITY=scatter ./runme 32
	KMP_AFFINITY=scatter ./runme 64
	KMP_AFFINITY=scatter ./runme 96
	KMP_AFFINITY=scatter ./runme 128
	KMP_AFFINITY=scatter ./runme 160
	KMP_AFFINITY=scatter ./runme 192
	KMP_AFFINITY=scatter ./runme 224
	KMP_AFFINITY=scatter ./runme 256

run-mic: runmeMIC
	ssh mic0 "LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC 32"
	ssh mic0 "LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC 64"
	ssh mic0 "LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC 96"
	ssh mic0 "LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC 128"
	ssh mic0 "LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC 160"
	ssh mic0 "LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC 224"
	ssh mic0 "LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC 256"
