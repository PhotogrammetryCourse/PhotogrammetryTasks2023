#include "gms_matcher.h"

#include "gms_matcher_impl.h"

// source: https://github.com/JiawangBian/GMS-Feature-Matcher
void phg::filterMatchesGMS(const std::vector <cv::DMatch> &matches_all,
                           const std::vector <cv::KeyPoint> kp1,
                           const std::vector <cv::KeyPoint> kp2,
                           const cv::Size &sz1,
                           const cv::Size &sz2,
                           std::vector <cv::DMatch> &matches_gms)
{
    matches_gms.clear();

    using namespace std;
    using namespace cv;

    std::vector<bool> vbInliers;
    gms_matcher gms(kp1, sz1, kp2, sz2, matches_all);
    int num_inliers = gms.GetInlierMask(vbInliers, true, true);
    cout << "Get total " << num_inliers << " matches." << endl;

    for (size_t i = 0; i < vbInliers.size(); ++i) {
        if (vbInliers[i]) {
            matches_gms.push_back(matches_all[i]);
        }
    }
}
