# Compiler: Use the GNU C++ compiler
CXX = g++

# Compiler flags:
# -g:        Include debug symbols
# -O3:       Enable high-level optimizations for performance
# -Wall:     Show all compiler warnings
# -fopenmp:  Enable OpenMP support
CXXFLAGS = -g -O3 -Wall -fopenmp -std=c++17

# Name of the final executable file
TARGET = benchmark

# Automatically find all .cpp files in the current directory
SOURCES = $(wildcard *.cpp)
# Create a list of object files (.o) from the source files
OBJECTS = $(SOURCES:.cpp=.o)

# The default rule: build the executable
all: $(TARGET)

# Rule to link all the object files into the final executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Rule to compile each .cpp source file into a .o object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to clean up compiled files
clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all clean