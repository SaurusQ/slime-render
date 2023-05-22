
CC := g++
CFLAGS := -Wall -Wextra -Ilib/glfw/include -I lib/glad/include
LIBS := -Llib/glfw/lib-mingw-w64 -lglfw3dll -lOpenGL32
SRC_DIR := src
BUILD_DIR := build
TARGET := slime.exe	

SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard lib/glad/src/glad.c)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean