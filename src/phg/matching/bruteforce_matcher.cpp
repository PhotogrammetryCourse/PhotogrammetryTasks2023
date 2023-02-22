#include "bruteforce_matcher.h"

#include <iostream>
#include <libutils/rasserts.h>


void phg::BruteforceMatcher::train(const cv::Mat &train_desc)
{
    if (train_desc.rows < 2) {
        throw std::runtime_error("BruteforceMatcher:: train : needed at least 2 train descriptors");
    }

    train_desc_ptr = &train_desc;
}

void phg::BruteforceMatcher::knnMatch(const cv::Mat &query_desc,
                                      std::vector<std::vector<cv::DMatch>> &matches,
                                      int k) const
{
    if (!train_desc_ptr) {
        throw std::runtime_error("BruteforceMatcher:: knnMatch : matcher is not trained");
    }

    if (k != 2) {
        throw std::runtime_error("BruteforceMatcher:: knnMatch : only k = 2 supported");
    }

    std::cout << "BruteforceMatcher::knnMatch : n query desc : " << query_desc.rows << ", n train desc : " << train_desc_ptr->rows << std::endl;

    const int ndesc = query_desc.rows;

    matches.resize(ndesc);

    const cv::Mat &train_desc = *train_desc_ptr;
    const int n_train_desc = train_desc.rows;

    #pragma omp parallel for
    for (int qi = 0; qi < ndesc; ++qi) {
        std::vector<cv::DMatch> &dst = matches[qi];
        dst.clear();
        dst.reserve(2);

        for (int ti = 0; ti < n_train_desc; ++ti) {
            cv::DMatch match;
            match.distance = cv::norm(train_desc.row(ti) - query_desc.row(qi), cv::NORM_L2);
            match.imgIdx = 0;
            match.queryIdx = qi;
            match.trainIdx = ti;
            if (dst.empty()) {
                dst.push_back(match);
            }
            else
            if (dst.size() == 1) {
                dst.push_back(match);
                if (dst[0].distance > dst[1].distance) {
                    std::swap(dst[0], dst[1]);
                }
            }
            else
            if (dst.size() == 2) {
                if (dst[0].distance > match.distance) {
                    dst[1] = dst[0];
                    dst[0] = match;
                }
                else
                if (dst[1].distance > match.distance) {
                    dst[1] = match;
                }
            }
            else {
                // если внутри openmp цикла бросить исключение, то программа упадет с сегфолтом
                // нужно либо перехватывать исключения и обрабатывать их вне цикла, либо оповещать об ошибках иначе
                std::cerr << "BruteforceMatcher:: knnMatch : invalid number of matches" << std::endl;
            }
        }
    }
}
