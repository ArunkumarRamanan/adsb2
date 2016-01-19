CXX = g++
CXXFLAGS = -g -O3 -std=c++11 -fopenmp
LDFLAGS += -fopenmp -L/usr/local/lib
#LDLIBS = -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -lboost_timer -lboost_chrono -lboost_thread -lboost_system -lboost_program_options -lboost_filesystem 
LDLIBS = -lopencv_imgproc -lopencv_highgui -lopencv_core -lboost_timer -lboost_chrono -lboost_thread -lboost_system -lboost_program_options -lboost_filesystem 
HEADERS =
COMMON = 


PROGS = heart

all:	$(PROGS)

$(PROGS):	%:      %.cpp $(HEADERS) $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp $(COMMON) $(LDLIBS)

