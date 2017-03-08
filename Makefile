CXX = g++
CXXFLAGS = -I. -std=c++14 -O3 -Wall -g -march=corei7-avx -mtune=corei7-avx

all: dot align

dot: main.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o dot-product main.cpp align.cpp

align: align_test.cpp align.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o align_test align_test.cpp align.cpp
