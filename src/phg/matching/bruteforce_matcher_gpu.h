#pragma once

#include "descriptor_matcher.h"

namespace phg {

    struct BruteforceMatcherGPU : DescriptorMatcher {

        void train(const cv::Mat &train_desc) override;

        void knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const override;

    private:

        const cv::Mat *train_desc_ptr = nullptr;
    };

}