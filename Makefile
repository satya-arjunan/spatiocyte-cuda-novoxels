VIS=\
		Visualizer

SRC=\
		Spatiocyte\
		Model\
		Stepper\
		Compartment\
		Species\
		Diffuser\
		Reaction\
		VisualLogger

IFLAGS = -I. 
LDFLAGS = # -L$(HOME)/root/lib -lRandom
#CXXFLAGS = -O3 -march=native -mavx -Werror -Wfatal-errors -Wall -std=c++0x #-fprofile-use #-pg -fprofile-generate
#CXXFLAGS = -O3 -march=core-avx2 -Wfatal-errors -Wall -std=c++0x #-fprofile-use #-pg -fprofile-generate
CXXFLAGS = -O3 -std=c++11 #-fprofile-use #-pg -fprofile-generate
#CXXEFLAGS = -arch=compute_53 -lcurand
CXXEFLAGS = -gencode arch=compute_60,code=sm_60 -lcurand -lcuda
CUFLAGS = -dc
#CXX = icc
#CXX = g++
CXX = g++
CUXX = nvcc
GUILIBS = $(shell pkg-config --libs gtkmm-2.4 gtkglextmm-x11-1.2 libpng)
GUIFLAGS = $(shell pkg-config --cflags gtkmm-2.4 gtkglextmm-x11-1.2) -I.
CPPFLAGS = -DG_DISABLE_DEPRECATED -DGDK_PIXBUF_DISABLE_DEPRECATED -DPNG_SKIP_SETJMP_CHECK
OBJECTS=${SRC:=.o}
SPATIOCYTE_CUDA = spatiocyte-cuda
VISUALIZER = visualizer


all: $(SPATIOCYTE_CUDA) $(VISUALIZER)

$(SPATIOCYTE_CUDA): $(OBJECTS)
		$(CUXX) $(CXXFLAGS) $(CXXEFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)

$(VISUALIZER):
		$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(IFLAGS) $(GUIFLAGS) -o $@ $(VIS).cpp $(GUILIBS)

%.o: %.cpp
		$(CUXX) $(CXXFLAGS) $(CXXEFLAGS) $(CUFLAGS) $(IFLAGS) -c -o $@ $<

%.o: %.cu
		$(CUXX) $(CXXFLAGS) $(CXXEFLAGS) $(CUFLAGS) $(IFLAGS) -c -o $@ $<

clean:
		rm -f $(SPATIOCYTE_CUDA) $(VISUALIZER) *.o
