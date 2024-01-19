#include "image.hpp"
#include "imageGPU.hpp"
#include "shaderHandler.hpp"
#include "definitions.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifdef GUI
    #include "UI.hpp"
#endif

#include <algorithm>
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>

ImgConfig imgConfig
{
    AgentConfig
    {
        1.0,
        90.0,
        30.0,
        9.0,
        0
    },
    0.007,
    1.0,
    false
};

bool showUI = false;
bool dragMouse = false;

float translateY = 0.0;
float translateX =  0.0;
float zoom = 1.0;

constexpr char wndName[] = "slime";

ShaderHandler* shaderHandlerPtr;

float vertices[] = {
    // Positions        // Texture Coordinates
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
     1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
     1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
    -1.0f,  1.0f, 0.0f, 0.0f, 1.0f
};

unsigned int indices[] = {
    0, 1, 2,
    2, 3, 0
};

void printMat(glm::mat4 matrix, std::string name)
{
    std::cout << name << std::endl;
    const float* ptr = glm::value_ptr(matrix);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << ptr[i + j * 4] << " ";
        }
        std::cout << std::endl;
    }
}

void updateViewMatrix()
{
    glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(translateX, translateY, 0.0f));
    glm::mat4 projection = glm::ortho(-1.0 * zoom, 1.0 * zoom, -1.0 * zoom, 1.0 * zoom, 1.0, -1.0);
    GLint shaderProgram = shaderHandlerPtr->getShaderProgramId();
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
}

void key_callback(GLFWwindow* wnd, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(wnd, true);
    if (key == GLFW_KEY_M && action == GLFW_PRESS)
        showUI = !showUI;
    if (key == GLFW_KEY_C && action == GLFW_PRESS)
        imgConfig.updateAgents = !imgConfig.updateAgents;
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        zoom = 1.0;
        translateX = 0.0;
        translateY = 0.0;
    }
}

void scroll_callback(GLFWwindow* wnd, double xoffset, double yoffset)
{
    zoom = std::max(0.01, zoom + yoffset * 0.04 * zoom);
}

void mouseButton_callback(GLFWwindow* wnd, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        dragMouse = true;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        dragMouse = false;
}

void cursor_callback(GLFWwindow* wnd, double xpos, double ypos)
{
    static double lasty = ypos;
    static double lastx = xpos;

    double diffx = xpos - lastx;
    double diffy = ypos - lasty;
    if (dragMouse && !showUI)
    {
        translateX += diffx * 0.001 * zoom;
        translateY -= diffy * 0.001 * zoom;
    }
    lasty = ypos;
    lastx = xpos;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void fpsHandler(double cur, GLFWwindow* window)
{
    static unsigned int nFrames = 0;
    static double last = glfwGetTime();
    nFrames++;
    if(cur > (last + 1.0))
    {
        float fps = nFrames;
        last += 1.0;

        std::stringstream ss;
        ss << wndName << " " << fps;
        glfwSetWindowTitle(window, ss.str().c_str());
        nFrames = 0;
    }
}

int main()
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(W_4K, H_4K, wndName, NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouseButton_callback);
    glfwSetCursorPosCallback(window, cursor_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGL())
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Shaders
    ShaderHandler shaderHandler;
    shaderHandler.addShader("shaders/fragment.frag", GL_FRAGMENT_SHADER);
    shaderHandler.addShader("shaders/vertex.vert", GL_VERTEX_SHADER);
    shaderHandler.link();
    shaderHandlerPtr = &shaderHandler;

    unsigned int VBO, VAO, EBO;
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

#ifdef GUI
    UI configUI(window);
#endif

    Image img{W_4K, H_4K};
    ImageGPU imgGPU{img, 100};
    GLuint texture = imgGPU.getTexture();
    imgGPU.activateCuda();
    img.setColor(RGB{0.0, 0.0, 0.0});
    //img.drawCircle(100, 1, 10, RGB{1.0, 0, 0});
    //img.randomize();
    imgGPU.update(img);
    imgGPU.deactivateCuda();
    
    std::vector<float> kernelData = {
        1.0 / 256,  4.0 / 256,  6.0 / 256,  4.0 / 256,  1.0 / 256,
        4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256,  4.0 / 256,
        6.0 / 256, 24.0 / 256, 36.0 / 256, 24.0 / 256,  6.0 / 256,
        4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256,  4.0 / 256,
        1.0 / 256,  4.0 / 256,  6.0 / 256,  4.0 / 256,  1.0 / 256
    };
    imgGPU.addConvKernel(2, kernelData);
    imgGPU.addConvKernel(1, std::vector<float>(9, 1.0 / 9.0));
    imgGPU.addConvKernel(3, std::vector<float>{
        0.2  / 9.0, 0.2  / 9.0, 0.2  / 9.0,
        0.2  / 9.0, 7.4  / 9.0, 0.2  / 9.0,
        0.2  / 9.0, 0.2  / 9.0, 0.2  / 9.0
    });

    imgGPU.configAgents(1000000, StartFormation::MIDDLE);
    imgGPU.configAgentParameters(imgConfig.ac);

    unsigned int IMG_W = img.getWidth();
    unsigned int IMG_H = img.getHeigth();

    double currentTime = glfwGetTime();

    while (!glfwWindowShouldClose(window))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        currentTime = glfwGetTime();

        fpsHandler(currentTime, window);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (imgConfig.updateAgents)
        {
            imgGPU.activateCuda();
            imgGPU.convolution(1, 1, imgConfig.diffuse);
            imgGPU.evaporate(imgConfig.evaporate);
            imgGPU.updateAgents();
            imgGPU.deactivateCuda();
        }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, imgGPU.getPbo());

        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width_, height_, 0, GL_RGB, GL_FLOAT, NULL);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, IMG_W, IMG_H, 0, GL_RGB, GL_FLOAT, 0);
        //glGenerateMipmap(GL_TEXTURE_2D);

        //glBindTexture(GL_TEXTURE_2D, texture);
        glActiveTexture(GL_TEXTURE0);

        // Render a VAO that cover the screen
        shaderHandler.use();
        updateViewMatrix();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

#ifdef GUI
        if (showUI)
        {
            configUI.update(window, imgConfig);
            imgGPU.configAgentParameters(imgConfig.ac);
        }
#endif

        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &EBO);

    glfwTerminate();
    return 0;
}

