#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdint.h>

#include "shaderHandler.hpp"

constexpr uint64_t scrWidth = 800;
constexpr uint64_t scrHeight = 600;

constexpr char windowName[] = "slime";

int main(void)
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(scrWidth, scrHeight, windowName, NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    //glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    ShaderHandler shaderHandler;
    shaderHandler.addShader("shaders/fragment.frag");
    shaderHandler.addShader("shaders/vertex.vert");
    shaderHandler.link();


    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}