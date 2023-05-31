#pragma once

#include "definitions.hpp"

#include <inttypes.h>
#include <memory>

class Image
{
public:
    Image(unsigned int width, unsigned int height);
    ~Image();
    void randomize();
    void setColor(RGB color);
    void drawCircle(unsigned int x, unsigned int y, unsigned int radius, RGB rgb);
    void colorOneByOne();
    const RGB* getPtr() const { return imagePtr_.get(); }
    unsigned int getWidth() const { return width_; }
    unsigned int getHeight() const { return height_; }
    unsigned int getBufferSize() const { return pixels_ * sizeof(RGB); }
    unsigned int getPaddedBufferSize(unsigned int padding) const { return (width_ + padding * 2) * (height_ + padding * 2) * sizeof(RGB); }
private:
    unsigned int width_;
    unsigned int height_;
    unsigned int pixels_;
    std::unique_ptr<RGB[]> imagePtr_;
};