#pragma once

#include <opencv2/core.hpp>

namespace phg {

    cv::Mat findHomography(const std::vector<cv::Point2f> &points_lhs,
                               const std::vector<cv::Point2f> &points_rhs);

    cv::Mat findHomographyCV(const std::vector<cv::Point2f> &points_lhs,
                           const std::vector<cv::Point2f> &points_rhs);

    cv::Point2d transformPoint(const cv::Point2d &pt, const cv::Mat &T);
    cv::Point2d transformPointCV(const cv::Point2d &pt, const cv::Mat &T);
}