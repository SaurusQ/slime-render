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

void UI::update(GLFWwindow*wnd, SimConfig& sc, SimUpdate& su)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Draw UI
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    ImGuiIO& io = ImGui::GetIO();

    {
        ImGui::Begin("Configuration");
        
        if (ImGui::CollapsingHeader("Agent config", ImGuiTreeNodeFlags_DefaultOpen))
        {
            su.agentSettings |= ImGui::DragFloat("speed", &sc.ac.speed, 1, 0.0, 100.0);
            su.agentSettings |= ImGui::SliderFloat("turn speed", &sc.ac.turnSpeed, 0.0, 1000.0);
            su.agentSettings |= ImGui::SliderFloat("sensor angle", &sc.ac.sensorAngleSpacing, 22.5, 45.0);
            su.agentSettings |= ImGui::SliderFloat("sensor offset", &sc.ac.sensorOffsetDst, 1.0, 50.0);
            su.agentSettings |= ImGui::SliderInt("sensor size", (int*)&sc.ac.sensorSize, 0, 10.0);
        }

        if (ImGui::CollapsingHeader("Population config", ImGuiTreeNodeFlags_DefaultOpen))
        {
            int idx = -1;
            if(ImGui::SliderFloat("Red",       &sc.agentShare[0], 0.0f, 1.0f)) idx = 0;
            if(ImGui::SliderFloat("Green",     &sc.agentShare[1], 0.0f, 1.0f)) idx = 1;
            if(ImGui::SliderFloat("Blue",      &sc.agentShare[2], 0.0f, 1.0f)) idx = 2;
            if(ImGui::SliderFloat("Alpha",     &sc.agentShare[3], 0.0f, 1.0f)) idx = 3;
            if (idx != -1)
            {
                su.populationShare = true;
                this->balanceShare(sc.agentShare[idx], sc.agentShare[(idx + 1) % 4], sc.agentShare[(idx + 2) % 4], sc.agentShare[(idx + 3) % 4]);
            }
            su.population |= ImGui::SliderInt("Particles",   &sc.numAgents, 1, 1000000);
        }

        if (ImGui::CollapsingHeader("Trail config", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderFloat("evaporate", &sc.evaporate, 0.001, 1);
            ImGui::SliderFloat("diffuse", &sc.diffuse, 0.0, 50.0);
            ImGui::SliderFloat("Trail weight", &sc.trailWeight, 0.0, 100.0);
        }
        if (sc.updateAgents)
        {
            if (ImGui::Button("Pause")) sc.updateAgents = false;
        }
        else
        {
            if (ImGui::Button("Run")) sc.updateAgents = true;
        }
        ImGui::SameLine();
        if(ImGui::Button("Clear")) su.clearImg = true;
        
        ImGui::Text("Reset Spawn");
        if(ImGui::Button("Random")) { sc.startFormation = StartFormation::RANDOM; su.spawn = true; }
        ImGui::SameLine();
        if(ImGui::Button("Middle")) { sc.startFormation = StartFormation::MIDDLE; su.spawn = true; }
        ImGui::SameLine();
        if(ImGui::Button("Circle")) { sc.startFormation = StartFormation::CIRCLE; su.spawn = true; }
        ImGui::SameLine();
        if(ImGui::Button("RCircle")) { sc.startFormation = StartFormation::RANDOM_CIRCLE; su.spawn = true; }

        ImGui::Checkbox("Clear on spawn", &sc.clearOnSpawn);

        ImGui::End();
    }

    // Render UI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void UI::balanceShare(float changed, float& a, float& b, float& c)
{
    float na, nb, nc;
    float target = 1.0f - changed;
    float total = a + b + c;
    if (total == 0.0f)
    {
        a = 1.0f;
        b = 1.0f;
        c = 1.0f;
        total = 3.0f;
    }
    a = (a / total) * target;
    b = (b / total) * target;
    c = (c / total) * target;
}
