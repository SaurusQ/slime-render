#include "shaderHandler.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

ShaderHandler::ShaderHandler()
{
    shaderProgram_ = glCreateProgram();
}

ShaderHandler::~ShaderHandler()
{
    
}

bool ShaderHandler::addShader(std::string filename)
{
    // Load shader from file
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string shaderSource = buffer.str();



    // Compile shader
    GLuint shader;
    int success;
    shader = glCreateShader(GL_VERTEX_SHADER);
    const char* shaderSourcePtr = shaderSource.c_str();
    glShaderSource(shader, 1, &shaderSourcePtr, NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        return false;
    }
    
    // Add shader
    glAttachShader(shaderProgram_, shader);
    shaders_.push_back(shader);
    return true;
}

void ShaderHandler::link()
{
    int success;
    glLinkProgram(shaderProgram_);

    glGetProgramiv(shaderProgram_, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram_, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        return;
    }

    for (auto s : shaders_)
    {
        glDeleteShader(s);
    }

}