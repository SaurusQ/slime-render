#include "image.hpp"

#include <random>

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
