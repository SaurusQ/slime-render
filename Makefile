
CC := nvcc
CFLAGS := -I src -I lib/glad/include
IFLAGS := -I lib/imgui -I lib/imgui/backends
LIBS := -lglfw -lGL
SRC_DIR := src
BUILD_DIR := build
BUILD_DIR_IMGUI := build/imgui
TARGET := slime.exe
IMGUI_DIR := lib/imgui

SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu) $(wildcard lib/glad/src/glad.c)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

IMGUI_SRCS := $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
IMGUI_OBJS := $(patsubst $(IMGUI_DIR)/%.cpp,$(BUILD_DIR_IMGUI)/%.o,$(IMGUI_SRCS))

all: $(BUILD_DIR) $(BUILD_DIR)/$(TARGET)

gui: $(BUILD_DIR_IMGUI) $(IMGUI_OBJS)
	mkdir -p $(BUILD_DIR_IMGUI)
	$(MAKE) all

run: all
	./build/slime.exe


$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR_IMGUI): $(BUILD_DIR)
	mkdir -p $(BUILD_DIR_IMGUI)
	mkdir -p $(BUILD_DIR_IMGUI)/backends

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

$(BUILD_DIR_IMGUI)/%.o:$(IMGUI_DIR)/%.cpp
	$(CXX) $(CFLAGS) $(IFLAGS) -c -o $@ $<

#(BUILD_DIR_IMGUI)/%.o:$(IMGUI_DIR)/backends/%.cpp
#	$(CXX) $(CFLAGS) $(IFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean run gui