#include "image.hpp"

#include <opencv2/opencv.hpp>

#include <random>

Image::Image(unsigned int width, unsigned int heigth)
    : width_(width), height_(heigth)
{
    cv::namedWindow(wndName, cv::WINDOW_AUTOSIZE);
    pixels_ = width_ * height_;
    imagePtr_ = std::make_unique<RGB[]>(pixels_);
    std::fill(imagePtr_.get(), imagePtr_.get() + pixels_, RGB{0, 0, 0});
}

Image::~Image()
{
    cv::destroyAllWindows();
}

void Image::display()
{
    cv::Mat imageMat(width_, height_, CV_8UC3, imagePtr_.get());
    cv::imshow(wndName, imageMat);
    cv::waitKey(0);
}

void Image::randomize()
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    for (int i = 0; i < pixels_; i++)
    {
        imagePtr_[i].r = dist(rng);
        imagePtr_[i].g = dist(rng);
        imagePtr_[i].b = dist(rng);
    }

}

const RGB* Image::getPtr() const
{
    return imagePtr_.get();
 
}