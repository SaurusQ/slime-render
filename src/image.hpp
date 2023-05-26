#pragma once

#include "definitions.hpp"

#include <inttypes.h>
#include <memory>

class Image
{
public:
    Image(unsigned int width = W_4K, unsigned int height = H_4K);
    ~Image();
    void randomize();
    const RGB* getPtr() const { return imagePtr_.get(); }
    unsigned int getWidth() const { return width_; }
    unsigned int getHeight() const { return height_; }
    unsigned int getBufferSize() const { return pixels_ * sizeof(RGB); }
    unsigned int getPaddedBufferSize(unsigned int padding) { return (width_ + padding) * (height_ + padding) * sizeof(RGB); }
private:
    unsigned int width_;
    unsigned int height_;
    unsigned int pixels_;
    std::unique_ptr<RGB[]> imagePtr_;
};