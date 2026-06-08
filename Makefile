CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pthread
PKG_CFLAGS := $(shell pkg-config --cflags opencv4 libcamera)
PKG_LIBS := $(shell pkg-config --libs opencv4 libcamera)

all: parkcam

parkcam: parkcam.cpp
	$(CXX) $(CXXFLAGS) $(PKG_CFLAGS) $< -o $@ $(PKG_LIBS)

clean:
	rm -f parkcam
