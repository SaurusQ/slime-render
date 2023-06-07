#include "image.hpp"

#include <random>
#include <math.h>

Image::Image(unsigned int width, unsigned int heigth)
{
    // Image only handles values with a factor of 32
    if (width % 32)  width_  = width + (32 - width % 32); // TODO let the image be any size, limit in kernel and correct block dims
    else             width_  = width;
    if (heigth % 32) height_ = heigth + (32 - heigth % 32);
    else             height_ = heigth;

    pixels_ = width_ * height_;
    imagePtr_ = std::make_unique<RGB[]>(pixels_);
    std::fill(imagePtr_.get(), imagePtr_.get() + pixels_, RGB{0, 0, 0});
}

Image::~Image()
{

}

void Image::randomize()
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (int i = 0; i < pixels_; i++)
    {
        imagePtr_[i].r = dist(rng);
        imagePtr_[i].g = dist(rng);
        imagePtr_[i].b = dist(rng);
    }

}

void Image::setColor(RGB color)
{
    for (int y = 0; y < height_; y++)
    {
        for (int x = 0; x < width_; x++)
        {
            imagePtr_.get()[(x + y * width_)] = color;
        }
    }
}

void Image::drawCircle(unsigned int cx, unsigned int cy, unsigned int radius, RGB rgb)
{
    unsigned int r2 = radius * radius;
    for (int y = 0; y < height_; y++)
    {
        for (int x = 0; x < width_; x++)
        {
            int dx = cx -x;
            int dy = cy -y;
            if(r2 >= dx * dx + dy * dy) imagePtr_.get()[(x + y * width_)] = rgb;
        }
    }
}

void Image::colorOneByOne()
{
    static unsigned int idx = 0;
    imagePtr_[idx].b = 1.0;
    idx++;
    if (idx == pixels_)
    {
        idx = 0;
    }
}
