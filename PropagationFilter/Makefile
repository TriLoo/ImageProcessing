
DBG = -O3 -march=native #-Wall

# Without openmp
# MP = 
# MP_DEFINE = 

# With openmp
MP = -fopenmp
MP_DEFINE = -DOPENMP

MEX    = mex
MEXDBG = #-g
MEXOPT =  -O
MEXCFLAGS = CXXFLAGS='$$CXXFLAGS $(DBG) $(MP)'
MEXLDFLAGS = LDFLAGS='$$LDFLAGS $(DBG) $(MP)'
MEXFLAGS = $(MEXDBG) $(MEXOPT) $(MEXCFLAGS) $(MEXLDFLAGS)


all: pfilter.mexa64 

pfilter.mexa64: pfilter.cpp pfilter-matrix.cpp
	$(MEX) $(MEXFLAGS) $(MP_DEFINE) -o pfilter pfilter.cpp pfilter-matrix.cpp 

clean:
	rm -f pfilter *.o *.mexa64

clean-all: clean
	rm -f *~ 



