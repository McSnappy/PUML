.PHONY = all debug mltest clean 

CXX = /usr/bin/g++
CXXFLAGS = -O2 -Wall -std=c++11
LDDFLAGS = -lpumlbrent -lpthread

UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
	CXXFLAGS += -ffloat-store
	LDDFLAGS += -Wl,-rpath=.
endif

all: clean mltest 

debug: CXXFLAGS += -g -DDEBUG=1
debug: clean mltest

mltest:
	$(CXX) $(CXXFLAGS) -fPIC -rdynamic -shared -x c++ brent/brent.c -o libpumlbrent.so
	$(CXX) $(CXXFLAGS) -L . mltest.cpp machinelearning.cpp mlutil.cpp logging.cpp knn.cpp kmeans.cpp decisiontree.cpp randomforest.cpp boosting.cpp cJSON/cJSON.cpp -o mltest $(LDDFLAGS)

clean: 
	rm -f ./libpumlbrent.so
	rm -f ./mltest
	rm -rf ./mltest.dSYM
