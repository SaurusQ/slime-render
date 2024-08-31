#pragma once

#include "definitions.hpp"

#include <string>
#include <vector>
#include <map>


#define OUT_FILE "current_config.json"

class ConfigReader
{
public:
    ConfigReader(std::string filepath);

    bool next(SimConfig& sc);

    void printOutConfig(const SimConfig& sc);

private:
    RGB hexToColor(std::string hex);
    std::string rgbToHex(RGB color);
    std::string colorsToHex(const RGB* colors);

    unsigned int currentFrame_ = 0;
    unsigned int endFrame_ = 0;

    SimConfig currentConfig_;

    std::map<unsigned int, SimConfig> configsF_;

};
