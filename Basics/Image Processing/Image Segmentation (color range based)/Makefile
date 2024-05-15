CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -fPIC -Iinclude $(shell pkg-config --cflags Qt5Widgets Qt5Gui Qt5Core)
LDFLAGS = $(shell pkg-config --libs Qt5Widgets Qt5Gui Qt5Core)

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INCLUDE_DIR = include
MOC_DIR = moc

TARGET = $(BIN_DIR)/my_program

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
HEADERS = $(wildcard $(INCLUDE_DIR)/*.h)
MOC_SRCS = $(HEADERS:$(INCLUDE_DIR)/%.h=$(MOC_DIR)/moc_%.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS)) $(patsubst $(MOC_DIR)/%.cpp,$(MOC_DIR)/%.o,$(MOC_SRCS))

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)
	@echo "Linking complete: $(TARGET)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled: $< -> $@"

$(MOC_DIR)/moc_%.cpp: $(INCLUDE_DIR)/%.h
	@mkdir -p $(MOC_DIR)
	moc $< -o $@

$(MOC_DIR)/%.o: $(MOC_DIR)/%.cpp
	@mkdir -p $(MOC_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "MOC Compiled: $< -> $@"

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(MOC_DIR)

.PHONY: all clean
