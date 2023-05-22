#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <vector>

class ShaderHandler
{
public:
    ShaderHandler();
    ~ShaderHandler();
    bool addShader(std::string filename);
    void link();
private:
    GLuint shaderProgram_;
    std::vector<GLuint> shaders_;
};
