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

    {
        ImGui::Begin("Configuration");
        
        if (ImGui::CollapsingHeader("Agent config", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::DragFloat("speed", &ic.ac.speed, 1, 0.0, 100.0);
            ImGui::SliderFloat("turn speed", &ic.ac.turnSpeed, 0.0, 180.0);
            ImGui::SliderFloat("sensor angle", &ic.ac.sensorAngleSpacing, 22.5, 45.0);
            ImGui::SliderFloat("sensor offset", &ic.ac.sensorOffsetDst, 1.0, 50.0);
            ImGui::SliderInt("sensor size", &sensorSize, 0, 10.0);

        }
        ImGui::SliderFloat("evaporate", &ic.evaporate, 0.001, 1);
        ImGui::SliderFloat("diffuse", &ic.diffuse, 0.0, 50.0);
        ImGui::SliderInt("Particles", &ic.numAgents, 1, 1000000);

        if (ic.updateAgents)
        {
            if (ImGui::Button("Pause")) ic.updateAgents = false;
        }
        else
        {
            if (ImGui::Button("Run")) ic.updateAgents = true;
        }
        ImGui::SameLine();
        if(ImGui::Button("Clear")) ic.clearImg = true;
        
        ImGui::Text("Reset Spawn");
        if(ImGui::Button("Random")) ic.startFormation = StartFormation::RANDOM; 
        ImGui::SameLine();
        if(ImGui::Button("Middle")) ic.startFormation = StartFormation::MIDDLE;
        ImGui::SameLine();
        if(ImGui::Button("Circle")) ic.startFormation = StartFormation::CIRCLE;

        ImGui::Checkbox("Clear on spawn", &ic.clearOnSpawn);

        ImGui::End();
    }

    ic.ac.sensorSize = sensorSize;

    // Render UI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
