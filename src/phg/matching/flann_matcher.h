#pragma once

#include "descriptor_matcher.h"
#include <opencv2/flann/miniflann.hpp>

namespace phg {

    struct FlannMatcher : DescriptorMatcher {

        FlannMatcher();

        void train(const cv::Mat &train_desc) override;

        void knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const override;

    private:

        std::shared_ptr<cv::flann::IndexParams> index_params;
        std::shared_ptr<cv::flann::SearchParams> search_params;
        std::shared_ptr<cv::flann::Index> flann_index;
    };

}
