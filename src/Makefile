.PHONY = all debug mltest clean 

CXX = /usr/bin/g++
CXXFLAGS = -O2 -Wall -std=c++11
LDDFLAGS = -lpthread

UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
	CXXFLAGS += 
	LDDFLAGS += -Wl,-rpath=.,--no-as-needed -pthread
endif

all: clean mltest 

debug: CXXFLAGS += -g -DDEBUG=1
debug: clean mltest

mltest:
	$(CXX) $(CXXFLAGS) -L . mltest.cpp mldata.cpp mlresults.cpp mlutil.cpp logging.cpp decisiontree.cpp randomforest.cpp -o mltest $(LDDFLAGS)

clean: 
	rm -f ./mltest
	rm -rf ./mltest.dSYM
