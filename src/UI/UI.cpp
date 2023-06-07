#include "UI.hpp"

UI::UI(GLFWwindow* wnd)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    ImGui_ImplGlfw_InitForOpenGL(wnd, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
}

UI::~UI()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UI::update(GLFWwindow*wnd, ImgConfig& ic)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Draw UI
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    ImGuiIO& io = ImGui::GetIO();

    int sensorSize = ic.ac.sensorSize;

    int counter = 0;
    {
        ImGui::Begin("Hello, world!");
        
        if (ImGui::CollapsingHeader("Agent config", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::DragFloat("speed", &ic.ac.speed, 1.0, 0.0, 10.0);
            ImGui::SliderFloat("turn speed", &ic.ac.turnSpeed, 0.0, 180.0);
            ImGui::SliderFloat("sensor angle", &ic.ac.sensorAngleSpacing, 22.5, 45.0);
            ImGui::SliderFloat("sensor offset", &ic.ac.sensorOffsetDst, 1.0, 50.0);
            ImGui::SliderInt("sensor size", &sensorSize, 0, 10.0);

        }
        ImGui::SliderFloat("evaporate", &ic.evaporate, 0.1, 0.001);
        ImGui::SliderFloat("diffuse", &ic.diffuse, 0.0, 1.0);

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::End();
    }

    ic.ac.sensorSize = sensorSize;

    // Render UI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
