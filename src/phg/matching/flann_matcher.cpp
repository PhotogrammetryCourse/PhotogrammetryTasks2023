#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    // я 100% не там искал, но вот здесь нашёл дефолтные параметры 4/32
    // http://lumiguide.github.io/haskell-opencv/doc/src/OpenCV-Features2d.html
    // и с ними как будто нормально работает
    index_params = flannKdTreeIndexParams(4);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices, dists;
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);

    int match_n = query_desc.rows;
    matches.resize(match_n);
    for (int i = 0; i < match_n; i++) {
        matches[i].resize(k);
        for (int j = 0; j < k; j++) {
            matches[i][j] = cv::DMatch(i, indices.at<int>(i, j), dists.at<float>(i, j));
        }
    }
}
