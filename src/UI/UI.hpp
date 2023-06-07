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

    void update(GLFWwindow* wnd, ImgConfig& ic);
private:

};
