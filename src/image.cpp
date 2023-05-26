#include "image.hpp"

#include <random>
#include <math.h>

Image::Image(unsigned int width, unsigned int heigth)
    : width_(width), height_(heigth)
{
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
    std::uniform_int_distribution<int> dist(0, 255);

    for (int i = 0; i < pixels_; i++)
    {
        imagePtr_[i].r = dist(rng); // TODO r and b swapped, float as a goal
        imagePtr_[i].g = dist(rng);
        imagePtr_[i].b = dist(rng);
    }

}

void Image::drawCircle(unsigned int x, unsigned int y, unsigned int radius, RGB rgb)
{
    unsigned int r2 = radius * radius;
    for (int y = 0; y < height_; y++)
    {
        for (int x = 0; x < width_; x++)
        {
            if(r2 >= x * x + y * y) imagePtr_[x + y * width_] = rgb;
        }
    }
}
