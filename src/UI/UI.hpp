#pragma once

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include "definitions.hpp"

class UI
{
public:
    UI(GLFWwindow* wnd);
    ~UI();

    void update(SimConfig& sc, SimUpdate& su, bool showConfig, bool showFps);

private:
    void updateConfig(SimConfig& sc, SimUpdate& su);
    void updateFps();

    void balanceShare(float changed, float& a, float& b, float& c);
};
