#include "UI.hpp"

UI::UI(GLFWwindow* wnd)
{
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(wnd, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");


}

UI::~UI()
{

}

void UI::update(GLFWwindow*wnd)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Information for the menu
    ImGui::Begin("test");

    ImGui::Text("test");

    ImGui::End();

    // Rendering
    ImGui::Render();
    int dW, dH;
    glfwGetFramebufferSize(wnd, &dW, &dH);
    glViewport(0, 0, dW, dH);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
