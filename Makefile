
CC := nvcc
CFLAGS := -I src -I lib/glad/include -allow-unsupported-compiler -I lib
IFLAGS := -I lib/imgui -I lib/imgui/backends -DGUI -I src/UI
LIBS := -lglfw -lGL
SRC_DIR := src
BUILD_DIR := build
BUILD_DIR_IMGUI := build/imgui
TARGET := slime.exe
IMGUI_DIR := lib/imgui

SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu) lib/glad/src/glad.c
IMGUI_SRCS := $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp

ifeq ($(filter gui,$(MAKECMDGOALS)),gui)
	CFLAGS += $(IFLAGS)
	SRCS += $(wildcard $(SRC_DIR)/UI/*.cpp) $(IMGUI_SRCS)
endif

OBJS := $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS:.cpp=.o)))
OBJS := $(OBJS:.cu=.o)
OBJS := $(OBJS:.c=.o)


all: $(BUILD_DIR) $(BUILD_DIR)/$(TARGET)

gui: test $(BUILD_DIR_IMGUI) all

test:
	echo $(MAKECMDGOALS)

run: all
	./build/slime.exe

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR_IMGUI): $(BUILD_DIR)
	mkdir -p $(BUILD_DIR_IMGUI)
	mkdir -p $(BUILD_DIR_IMGUI)/backends
	mkdir -p build/UI

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: lib/glad/src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

ifeq ($(filter gui,$(MAKECMDGOALS)),gui)
$(BUILD_DIR)/%.o: $(SRC_DIR)/UI/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@
endif

$(BUILD_DIR)/%.o : $(IMGUI_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o : $(IMGUI_DIR)/backends/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)


$(BUILD_DIR_IMGUI)/%.o:$(IMGUI_DIR)/%.cpp
	$(CXX) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR_IMGUI)/%.o:$(IMGUI_DIR)/backends/%.cpp
	$(CXX) $(CFLAGS) $(IFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean run gui