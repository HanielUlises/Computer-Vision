CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -pedantic
LDFLAGS =

OBJ = main.o image_handler.o Utils.o

DEPS = image_handler.h Utils.h

TARGET = prediction

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

clean:
	rm -f $(OBJ) $(TARGET)
