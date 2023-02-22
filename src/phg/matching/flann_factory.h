#pragma once

#include <opencv2/flann/miniflann.hpp>
#include <opencv2/flann/kdtree_index.h>


namespace phg {

    inline std::shared_ptr<cv::flann::IndexParams> flannKdTreeIndexParams(int ntrees)
    {
        return std::make_shared<cv::flann::KDTreeIndexParams>(ntrees);
    }

    inline std::shared_ptr<cv::flann::SearchParams> flannKsTreeSearchParams(int nchecks)
    {
        return std::make_shared<cv::flann::SearchParams>(nchecks);
    }

    inline std::shared_ptr<cv::flann::Index> flannKdTreeIndex(const cv::Mat &data, const std::shared_ptr<cv::flann::IndexParams> &params)
    {
        return std::make_shared<cv::flann::Index>(data, *params);
    }
}
