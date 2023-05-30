#include "image.hpp"
#include "imageKernel.cuh"
#include "shaderHandler.hpp"
#include "definitions.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <iostream>

constexpr char wndName[] = "slime";

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

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
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

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Shaders
    ShaderHandler shaderHandler;
    shaderHandler.addShader("shaders/fragment.frag", GL_FRAGMENT_SHADER);
    shaderHandler.addShader("shaders/vertex.vert", GL_VERTEX_SHADER);
    shaderHandler.link();

    // OpenGL buffers
    /*GLuint renderbufferID;
    glGenRenderbuffers(1, &renderbufferID);
    glBindRenderbuffer(GL_RENDERBUFFER, renderbufferID);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
    // Register the buffer with cuda
    cudaGraphicsResource* cudaResource;
    cudaGraphicsGLRegisterImage(&cudaResource, renderbufferID, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsWriteDiscard);
    // Get pointer to the image
    uchar4* d_image;
    size_t imagePitch;
    cudaGraphicsMapResources(1, &cudaResource);
    cudaGraphicsResourceGetMappedPointer((void**)&d_image, &imagePitch, cudaResource);*/

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

    


    // Depth buffer
    //GLuint depthBuffer;
    //glGenRenderbuffers( 1, &depthBuffer );
    //glBindRenderbuffer( GL_RENDERBUFFER, depthBuffer );
    //glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height );
    // Unbind the depth buffer
    //glBindRenderbuffer( GL_RENDERBUFFER, 0 );

    // Frame buffer
    //GLuint framebuffer;
    //glGenFramebuffers( 1, &framebuffer );
    //glBindFramebuffer( GL_FRAMEBUFFER, framebuffer );
    //glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0 );
    //glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer );

    Image img{W_4K, H_4K};
    ImageKernel imgKernel{img, 100};
    GLuint texture = imgKernel.getTexture();
    img.drawCircle(1000, 1000, 500 , RGB{255, 0, 0});
    imgKernel.update(img);    

    std::vector<float> kernelData(25, 1.0 / 15.0);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        std::cout << "Loop" << std::endl;
        processInput(window);

        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //img.randomize();
        //imgKernel.update(img);
        imgKernel.convolution(2, kernelData);

        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W_4K, H_4K, 0, GL_RGB, GL_UNSIGNED_BYTE, img.getPtr());
        //glGenerateMipmap(GL_TEXTURE_2D);

        glBindTexture(GL_TEXTURE_2D, texture);
        glActiveTexture(GL_TEXTURE0);

        // Render a textured quad that covers the entire window
        shaderHandler.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &EBO);

    glfwTerminate();
    return 0;
}


/*static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

int main()
{
    cudaSetDevice(0);

    float* test = new float[2000];
    float* testGPU = nullptr;

    std::fill(test, test + 2000, 0);


    //cudaMalloc((void**)&testGPU, 10000 * sizeof(float));
    CHECK(cudaMalloc((void**)&testGPU, 2000 * sizeof(float)));

    CHECK(cudaMemcpy(testGPU, test, 2000 * sizeof(float), cudaMemcpyHostToDevice));

    kernel<<<1, 1>>>(2, 2, 2, testGPU);

    CHECK(cudaMemcpy(test, testGPU, 2000 * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(testGPU);
    delete[] test; 

    return 0;
}*/
