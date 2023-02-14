#pragma once

#include <opencv2/core.hpp>

cv::Mat concatenateImagesLeftRight(const cv::Mat &img0, const cv::Mat &img1);

std::string getTestName();

std::string getTestSuiteName();
