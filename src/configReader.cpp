#include "configReader.hpp"
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>

using json = nlohmann::json;

ConfigReader::ConfigReader(std::string filepath)
{
    std::ifstream file(filepath);

    if(!file.is_open())
    {
        std::cerr << "failed to open:" << filepath << std::endl;
        return;
    }

    json jsonData;
    try {
        file >> jsonData;
    } catch (json::parse_error& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return;
    }
    file.close();

    std::cout << "Parsed JSON data:" << std::endl;
    std::cout << jsonData.dump(4) << std::endl;

    SimConfig sc;
    for (const auto& su : jsonData["simconfig"])
    {
        unsigned int frame = su["frame"].get<int>();
        endFrame_ = std::max(frame, endFrame_);
        if (su.contains("species"))
        {
            const auto& species = su["species"];
            unsigned int i = 0;
            for (const auto& s : species)
            {
                if (s.contains("speed"))        sc.aConfigs[i].speed                = s["speed"].get<float>();
                if (s.contains("turn_speed"))   sc.aConfigs[i].turnSpeed            = s["turn_speed"].get<float>();
                if (s.contains("sensor_angle")) sc.aConfigs[i].sensorAngleSpacing   = s["sensor_angle"].get<float>();
                if (s.contains("sensor_dist"))  sc.aConfigs[i].sensorOffsetDst      = s["sensor_dist"].get<float>();
                i++;
                if (i == DIFFERENT_SPECIES) break;
            }
        }
        if (su.contains("colors"))
        {
            std::string colorData = su["colors"].get<std::string>();

            std::array<AgentColor, DIFFERENT_SPECIES> resultColors;
            std::istringstream iss(colorData);
            std::string scolor;

            unsigned int i = 0;
            while (iss >> scolor && i < DIFFERENT_SPECIES) {
                sc.aColors[i] = hexToColor(scolor);
                i++;
            }
        }
        if (su.contains("population"))
        {
            const auto& population = su["population"];
            if (population.contains("num_agents")) sc.numAgents = population["num_agents"].get<int>();
            if (population.contains("agent_share") && population["agent_share"].is_array())
            {
                unsigned int idx = 0;
                float total = 0.0f;
                float shares[DIFFERENT_SPECIES] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (const auto& share : population["agent_share"])
                {
                    float s = share.get<float>();
                    total += s;
                    shares[idx] == s;
                    idx++;
                    if (idx == DIFFERENT_SPECIES) break;
                }
                for (int i; i < DIFFERENT_SPECIES; i++)
                {
                    sc.agentShare[i] = shares[i] / total; // Balance to total of 1.0
                }

            }
        }
        if (su.contains("trail"))
        {
            const auto& trail = su["trail"];
            if (trail.contains("evaporate"))    sc.evaporate    = su["evaporate"].get<float>();
            if (trail.contains("diffuse"))      sc.diffuse      = su["diffuse"].get<float>();
            if (trail.contains("trail_weight")) sc.trailWeight  = su["trail_weight"].get<float>();
        }
        configsF_[frame] = sc;
        auto it = configsF_.find(0);
        if (it == configsF_.end())
        {
            configsF_[0] = sc;
            std::cout << "WARNING: json was not configured with zero frame for simconfig" << std::endl;
        }
    }
}

bool ConfigReader::next(SimConfig& sc, AgentColor* ac)
{
    bool end = false;
    while(currentFrame_ <= endFrame_ && !end)
    {
        auto configIt = configsF_.find(currentFrame_);
        if (configIt != configsF_.end())
        {
            end = true;
            const SimConfig& scu = configIt->second;
            bool updateAgents = sc.updateAgents;
            bool clearOnSpawn = sc.clearOnSpawn;
            StartFormation sf = sc.startFormation;

            sc = scu;

            // Keep some things constant
            sc.updateAgents = updateAgents;
            sc.clearOnSpawn = clearOnSpawn;
            sc.startFormation = sf;
        }
        currentFrame_++;
    }
    return end;
}

void ConfigReader::printOutConfig(const SimConfig& sc)
{
    std::ofstream file(OUT_FILE);

    if (!file.is_open())
    {
        std::cout << "Couldn't open output file" << std::endl;
        return;
    }
    
    json jsonObj;
    jsonObj["frame"] = 0;
    jsonObj["species"] = json::array();
    jsonObj["colors"] = colorsToHex(sc.aColors);

    for (int i = 0; i < DIFFERENT_SPECIES; i++)
    {
        jsonObj["species"].push_back({
            {"speed",           sc.aConfigs[i].speed},
            {"turn_speed",      sc.aConfigs[i].turnSpeed},
            {"sensor_angle",    sc.aConfigs[i].sensorAngleSpacing},
            {"sensor_dist",     sc.aConfigs[i].sensorOffsetDst},
        });
    }

    json jsonShares = json::array();
    for (float share : sc.agentShare)
    {
        jsonShares.push_back(share);
    }
    jsonObj["population"] = {
        {"num_agents", sc.numAgents},
        {"agent_share", jsonShares}
    };

    jsonObj["trail"] = {
        {"evaporate", sc.evaporate},
        {"diffuse", sc.diffuse},
        {"trail_weight", sc.trailWeight}
    };

    json jsonRes = {"simconfig", json::array()};
    jsonRes["simconfig"].push_back(jsonObj);

    file << jsonRes.dump(4);

    file.close();
}

RGB ConfigReader::hexToColor(std::string hex)
{
    std::string hexClean = hex;
    if (hexClean[0] == '#') {
        hexClean.erase(0, 1);
    }
    
    if (hexClean.length() != 6)
    {
        std::cerr << "Invalid color config: " << hex << std::endl;
        return RGB{1.0f, 1.0f, 1.0f};
    }

    std::stringstream ss;
    unsigned int r, g, b;
    ss << std::hex << hexClean.substr(0, 2);
    ss >> r;
    ss.clear();
    ss << std::hex << hexClean.substr(2, 2);
    ss >> g;
    ss.clear();
    ss << std::hex << hexClean.substr(4, 2);
    ss >> b;

    RGB res;
    res.r = std::min(1.0f, r / 255.0f);
    res.g = std::min(1.0f, g / 255.0f);
    res.b = std::min(1.0f, b / 255.0f);

    return res;
}

std::string ConfigReader::rgbToHex(RGB color)
{
    int r = static_cast<int>(color.r * 255.0f + 0.5f);
    int g = static_cast<int>(color.g * 255.0f + 0.5f);
    int b = static_cast<int>(color.b * 255.0f + 0.5f);

    r = std::max(0, std::min(255, r));
    g = std::max(0, std::min(255, g));
    b = std::max(0, std::min(255, b));

    std::ostringstream oss;
    oss << "#" << std::setw(2) << std::setfill('0') << std::hex << r
        << std::setw(2) << std::setfill('0') << std::hex << g
        << std::setw(2) << std::setfill('0') << std::hex << b;

    return oss.str();
}

std::string ConfigReader::colorsToHex(const RGB* colors)
{
    std::string res = "";
    for (int i = 0; i < DIFFERENT_SPECIES; i++)
    {
        res += rgbToHex(colors[i]);
        if (i != DIFFERENT_SPECIES - 1) res += " ";
    }
    return res;
}