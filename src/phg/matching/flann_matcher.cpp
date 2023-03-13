#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    int n = query_desc.rows;
    cv::Mat indices(n, k, CV_32SC1);
    cv::Mat distances(n, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices, distances, k, *search_params);

    matches.resize(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            matches[i].emplace_back(i, indices.at<int>(i, j), distances.at<float>(i, j));
        }
    }
}
