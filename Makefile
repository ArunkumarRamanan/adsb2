CXX = g++
CXXFLAGS = -g -O3 -std=c++11 -fopenmp -I/opt/caffe-fcn/include -Icaffex-fcn
LDFLAGS += -fopenmp -L/opt/caffe-fcn/lib -static
#LDLIBS = -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -lboost_timer -lboost_chrono -lboost_thread -lboost_system -lboost_program_options -lboost_filesystem 
LDLIBS =  -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive \
	 -lopencv_ml -lopencv_imgproc -lopencv_highgui -lopencv_core \
	 -lboost_timer -lboost_chrono -lboost_thread -lboost_filesystem -lboost_system -lboost_program_options \
	 -lprotoc -lprotobuf -lglog -lgflags -lleveldb -llmdb \
	 -lhdf5_hl -lhdf5 \
	 -ljpeg -lpng -ltiff -lgif -ljasper  \
	 -lsnappy -lz \
	 -lopenblas \
	 -lunwind -lrt -lm -ldl
	 
HEADERS = adsb2.h
COMMON = detector-caffe.o caffex-fcn/caffex.o


PROGS = heart

all:	$(PROGS)

$(PROGS):	%:      %.o $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp $(COMMON) $(LDLIBS)

%.o:	%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $*.cpp

