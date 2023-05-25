#pragma once

#include <inttypes.h>
#include <memory>

constexpr unsigned int width4k  = 3124;
constexpr unsigned int height4k = 2130;

struct RGB
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

class Image
{
public:
    Image(unsigned int width = width4k, unsigned int height = height4k);
    ~Image();
    void randomize();
    const RGB* getPtr() const { return imagePtr_.get(); }
    unsigned int getWidth() const { return width_; }
    unsigned int getHeight() const { return height_; }
private:
    unsigned int width_;
    unsigned int height_;
    unsigned int pixels_;
    std::unique_ptr<RGB[]> imagePtr_;
};