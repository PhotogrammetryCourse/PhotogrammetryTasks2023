#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    // TODO
    index_params = flannKdTreeIndexParams(2);
    search_params = flannKsTreeSearchParams(60);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices, dists;
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);

    for (int i = 0; i < query_desc.rows; i++) {
        std::vector<cv::DMatch> curr_matches;
        for (int j = 0; j < k; j++) {
            curr_matches.push_back(cv::DMatch(i, indices.at<int>(i, j), dists.at<float>(i, j)));
        }
        matches.push_back(curr_matches);
    }
}
