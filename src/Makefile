.PHONY = all debug mltest clean 

CXX = /usr/bin/g++
CXXFLAGS = -O2 -Wall -std=c++11

UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
	CXXFLAGS += -ffloat-store
endif

all: clean mltest 

debug: CXXFLAGS += -g -DDEBUG=1
debug: clean mltest

mltest:
	$(CXX) $(CXXFLAGS) -I . mltest.cpp machinelearning.cpp mlutil.cpp logging.cpp knn.cpp decisiontree.cpp randomforest.cpp boosting.cpp cJSON/cJSON.cpp -o mltest -lpthread

clean: 
	rm -f ./mltest
	rm -rf ./mltest.dSYM
