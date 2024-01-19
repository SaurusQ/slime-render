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
    bool addShader(std::string filename, int shaderType);
    void link();
    void use();
    GLuint getShaderProgramId() const { return shaderProgram_; }
private:
    GLuint shaderProgram_;
};
