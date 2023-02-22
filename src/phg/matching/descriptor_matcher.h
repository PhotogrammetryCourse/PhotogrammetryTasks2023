#pragma once

#include <opencv2/core.hpp>

namespace phg {

    struct DescriptorMatcher {

        virtual void train(const cv::Mat &train_desc) = 0;
        virtual void knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const = 0;

        static void filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches, std::vector<cv::DMatch> &filtered_matches);

        static void filterMatchesClusters(const std::vector<cv::DMatch> &matches,
                                          const std::vector<cv::KeyPoint> keypoints_query,
                                          const std::vector<cv::KeyPoint> keypoints_train,
                                          std::vector<cv::DMatch> &filtered_matches);
    };

}