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

bool ShaderHandler::addShader(std::string filename, int shaderType)
{
    // Load shader from file
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string shaderSource = buffer.str();



    // Compile shader
    GLuint shader;
    int success;
    shader = glCreateShader(shaderType);
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
    glDeleteShader(shader);
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
}

void ShaderHandler::use() 
{ 
    glUseProgram(shaderProgram_);
}
