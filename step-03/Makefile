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
	KMP_AFFINITY=scatter ./runme

run-mic: runmeMIC
	ssh mic0 "LD_LIBRARY_PATH=$(MIC_LD_LIBRARY_PATH) $(PWD)/runmeMIC"
