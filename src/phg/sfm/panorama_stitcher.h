#pragma once

#include <opencv2/core.hpp>

namespace phg {

    cv::Mat stitchPanorama(
            const std::vector<cv::Mat> &imgs,
            const std::vector<int> &parent,
            std::function<cv::Mat(const cv::Mat&, const cv::Mat&)> &homography_builder
    );

}