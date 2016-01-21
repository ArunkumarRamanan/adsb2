CXX = g++
CXXFLAGS = -g -O3 -std=c++11 -fopenmp -I/home/wdong/src/caffe-fcn/include -I/opt/chain/include
LDFLAGS += -fopenmp -L/home/wdong/src/caffe-fcn/lib -L/opt/chain/lib
#LDLIBS = -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -lboost_timer -lboost_chrono -lboost_thread -lboost_system -lboost_program_options -lboost_filesystem 
LDLIBS = caffex.o \
 	 -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive \
	 -lopencv_ml -lopencv_imgproc -lopencv_highgui -lopencv_core \
	 -lboost_timer -lboost_chrono -lboost_thread -lboost_filesystem -lboost_system -lboost_program_options \
	 -lprotoc -lprotobuf -lglog -lgflags -lleveldb -llmdb \
	 -lhdf5_hl -lhdf5 \
	 -ljpeg -lpng -ltiff -lgif -ljasper  \
	 -lsnappy -lz \
	 -lopenblas \
	 -lunwind -lrt -lm -ldl
	 
HEADERS = heart.h
COMMON = detector-caffe.o


PROGS = detect #heart detect

all:	$(PROGS)

$(PROGS):	%:      %.cpp $(HEADERS) $(COMMON)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $*.cpp $(COMMON) $(LDLIBS)

