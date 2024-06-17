CXX = g++
CXXFLAGS = -std=c++20 -fPIC -Iinclude $(shell pkg-config --cflags Qt5Widgets Qt5Gui Qt5Core opencv4) -Wno-deprecated-enum-enum-conversion
LDFLAGS = $(shell pkg-config --libs Qt5Widgets Qt5Gui Qt5Core opencv4)

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INCLUDE_DIR = include

TARGET = $(BIN_DIR)/knn_app

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
