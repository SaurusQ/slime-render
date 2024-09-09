#include "image.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <random>
#include <math.h>
#include <chrono>
#include <sstream>
#include <iomanip>

Image::Image(unsigned int width, unsigned int heigth)
    : width_(width), height_(heigth) 
{
    pixels_ = width_ * height_;
    imagePtr_ = std::make_unique<RGBA[]>(pixels_);
    std::fill(imagePtr_.get(), imagePtr_.get() + pixels_, RGBA{0.0f, 0.0f, 0.0f, 1.0f});
}

Image::~Image()
{

}

void Image::toFile() const
{
    RGBA* threadImg = new RGBA[width_ * height_];

    std::copy(imagePtr_.get(), imagePtr_.get() + (width_ * height_), threadImg);

    auto saveToFile = [this, threadImg]() {
        unsigned char* imageData = new unsigned char[width_ * height_ * 4];

        for (int i = 0; i < width_ * height_; ++i) {
            float r = threadImg[i].r;
            float g = threadImg[i].g;
            float b = threadImg[i].b;
            float a = threadImg[i].a;

            // Convert float [0.0, 1.0] to unsigned char [0, 255]
            imageData[i * 4 + 0] = static_cast<unsigned char>(r * 255.0f);
            imageData[i * 4 + 1] = static_cast<unsigned char>(g * 255.0f);
            imageData[i * 4 + 2] = static_cast<unsigned char>(b * 255.0f);
            imageData[i * 4 + 3] = static_cast<unsigned char>(a * 255.0f);
        }

        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm now_tm = *std::localtime(&now_c);

        std::stringstream filename;
        filename << "sim_";
        filename << std::put_time(&now_tm, "%Y-%m-%d_%H-%M-%S");
        filename << ".png";

        stbi_write_png(filename.str().c_str(), width_, height_, 4, imageData, width_ * 4);

        delete[] threadImg;
        delete[] imageData;
    };

    std::thread thread(saveToFile);
    thread.detach();
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

void Image::setColor(RGBA color)
{
    for (int y = 0; y < height_; y++)
    {
        for (int x = 0; x < width_; x++)
        {
            imagePtr_.get()[(x + y * width_)] = color;
        }
    }
}

void Image::drawCircle(unsigned int cx, unsigned int cy, unsigned int radius, RGBA rgb)
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
