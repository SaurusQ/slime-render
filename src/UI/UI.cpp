#include "UI.hpp"

#include <string>

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

void UI::update(SimConfig& sc, SimUpdate& su, bool showConfig, bool showFps)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (showConfig) updateConfig(sc, su);
    if (showFps) updateFps();

    // Render UI
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void UI::updateConfig(SimConfig& sc, SimUpdate& su)
{
    // Draw UI
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    ImGuiIO& io = ImGui::GetIO();

    {
        ImGui::Begin("Configuration");
        
        if(ImGui::CollapsingHeader("Agent config", ImGuiTreeNodeFlags_DefaultOpen))
        {
            for (int i = 0; i < DIFFERENT_SPECIES; i++)
            {
                std::string headerLabel = "Agent config: " + std::string(agentNames[i]);
                if (ImGui::CollapsingHeader(headerLabel.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
                {
                    su.agentSettings |= ImGui::DragFloat  ((std::string(agentNames[i]) + " speed").c_str(),          &sc.aConfigs[i].speed, 1, 0.0, 1000.0);
                    su.agentSettings |= ImGui::SliderFloat((std::string(agentNames[i]) + " turn speed").c_str(),     &sc.aConfigs[i].turnSpeed, 0.0, 1000.0);
                    su.agentSettings |= ImGui::SliderFloat((std::string(agentNames[i]) + " sensor angle").c_str(),   &sc.aConfigs[i].sensorAngleSpacing, 5.0, 90.0); // 22.5, 45.0
                    su.agentSettings |= ImGui::SliderFloat((std::string(agentNames[i]) + " sensor offset").c_str(),  &sc.aConfigs[i].sensorOffsetDst, 1.0, 50.0);
                    //su.agentSettings |= ImGui::SliderInt  ((std::string(agentNames[i]) + " sensor size").c_str(),    (int*)&sc.aConfigs[i].sensorSize, 0, 10.0);
                }
            }
            if (ImGui::CollapsingHeader("Agent colors", ImGuiTreeNodeFlags_DefaultOpen))
            {
                for (int i = 0; i < DIFFERENT_SPECIES; i++)
                {
                        su.agentSettings |= ImGui::ColorEdit3(agentNames[i], reinterpret_cast<float*>(&sc.aColors[i]),
                            ImGuiColorEditFlags_DisplayHex
                        );
                }
            }
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
}

void UI::updateFps()
{
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(100, 20), ImGuiCond_Always);

    ImGui::Begin("fps", nullptr,
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoBackground
    );

    ImGuiIO& io = ImGui::GetIO();
    float fps = io.Framerate;
    ImGui::Text("FPS: %.1f", fps);

    ImGui::End();
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
