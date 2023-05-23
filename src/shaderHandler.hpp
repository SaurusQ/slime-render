#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>

class ShaderHandler
{
public:
    ShaderHandler();
    ~ShaderHandler();
    bool addShader(std::string filename, int shaderType);
    void link();
    void use();
private:
    GLuint shaderProgram_;
};
