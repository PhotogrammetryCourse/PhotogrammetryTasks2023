#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <phg/matching/bruteforce_matcher.h>
#include <phg/matching/bruteforce_matcher_gpu.h>
#include <phg/sfm/homography.h>
#include <phg/matching/flann_matcher.h>
#include <phg/sift/sift.h>
#include <libutils/timer.h>
#include <phg/sfm/panorama_stitcher.h>
#include <phg/matching/gms_matcher.h>


#include "utils/test_utils.h"


// TODO enable both toggles for testing custom detector & matcher
#define ENABLE_MY_DESCRIPTOR 0
#define ENABLE_MY_MATCHING 1
#define ENABLE_GPU_BRUTEFORCE_MATCHER 0

#if ENABLE_MY_MATCHING
const double max_keypoints_rmse_px = 1.0;
#else
const double max_keypoints_rmse_px = 10.0;
#endif

const double max_color_rmse_8u = 20;

#define GAUSSIAN_NOISE_STDDEV 1.0


namespace {

    void drawMatches(const cv::Mat &img1,
                     const cv::Mat &img2,
                     const std::vector<cv::KeyPoint> &keypoints1,
                     const std::vector<cv::KeyPoint> &keypoints2,
                     const std::vector<cv::DMatch> &matches,
                     const std::string &path)
    {
        cv::Mat img_matches;
        drawMatches( img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1),
                     cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        cv::imwrite(path, img_matches);
    }

    cv::Mat getHomography(const cv::Mat &img1, const cv::Mat &img2)
    {
        using namespace cv;

        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );

        std::vector< std::vector<DMatch> > knn_matches;

#if ENABLE_MY_MATCHING
        phg::FlannMatcher matcher;
        matcher.train(descriptors2);
        matcher.knnMatch(descriptors1, knn_matches, 2);
#else
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
#endif

        std::vector<DMatch> good_matches(knn_matches.size());
        for (int i = 0; i < (int) knn_matches.size(); ++i) {
            good_matches[i] = knn_matches[i][0];
        }

#if ENABLE_MY_MATCHING
        phg::DescriptorMatcher::filterMatchesRatioTest(knn_matches, good_matches);
        {
            std::vector<DMatch> tmp;
            phg::DescriptorMatcher::filterMatchesClusters(good_matches, keypoints1, keypoints2, tmp);
            std::swap(tmp, good_matches);
        }
#else
        {
            std::vector<DMatch> tmp;
            phg::filterMatchesGMS(good_matches, keypoints1, keypoints2, img1.size(), img2.size(), tmp);
            std::swap(tmp, good_matches);
        }
#endif

        std::vector<cv::Point2f> points1, points2;
        for (const cv::DMatch &match : good_matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }

#if ENABLE_MY_MATCHING
        cv::Mat H = phg::findHomography(points1, points2);
#else
        cv::Mat H = phg::findHomographyCV(points1, points2);
#endif

        return H;
    }

    void evaluateStitching(const cv::Mat &img1, const cv::Mat &img2, double &keypoints_rmse, double &color_rmse,
                           const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                           const cv::Mat &descriptors1, const cv::Mat &descriptors2)
    {
        using namespace cv;

        std::vector<std::vector<DMatch>> knn_matches;

#if ENABLE_MY_MATCHING
        phg::FlannMatcher matcher;
        matcher.train(descriptors2);
        matcher.knnMatch(descriptors1, knn_matches, 2);
#else
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
#endif

        std::vector<DMatch> good_matches(knn_matches.size());
        for (int i = 0; i < (int) knn_matches.size(); ++i) {
            good_matches[i] = knn_matches[i][0];
        }

#if ENABLE_MY_MATCHING
        phg::DescriptorMatcher::filterMatchesRatioTest(knn_matches, good_matches);
        {
            std::vector<DMatch> tmp;
            phg::DescriptorMatcher::filterMatchesClusters(good_matches, keypoints1, keypoints2, tmp);
            std::swap(tmp, good_matches);
        }
#else
        {
            std::vector<DMatch> tmp;
            phg::filterMatchesGMS(good_matches, keypoints1, keypoints2, img1.size(), img2.size(), tmp);
            std::swap(tmp, good_matches);
        }
#endif

        std::vector<cv::Point2f> points1, points2;
        for (const cv::DMatch &match : good_matches) {
            points1.push_back(keypoints1[match.queryIdx].pt);
            points2.push_back(keypoints2[match.trainIdx].pt);
        }
#if ENABLE_MY_MATCHING
        cv::Mat H = phg::findHomography(points1, points2);
#else
        cv::Mat H = phg::findHomographyCV(points1, points2);
#endif

        if (good_matches.size() < 4) {
            throw std::runtime_error("too few matches");
        }

        keypoints_rmse = 0;
        for (int i = 0; i < (int) good_matches.size(); ++i) {
#if ENABLE_MY_MATCHING
            cv::Point2f pt = phg::transformPoint(points1[i], H);
#else
            cv::Point2f pt = phg::transformPointCV(points1[i], H);
#endif
            cv::Point2f diff = pt - points2[i];
            keypoints_rmse += diff.x * diff.x + diff.y * diff.y;
        }
        keypoints_rmse /= good_matches.size();
        keypoints_rmse = std::sqrt(keypoints_rmse);

        color_rmse = 0;
        int64_t count = 0;
        for (int y = 0; y < img1.rows; ++y) {
            for (int x = 0; x < img1.cols; ++x) {
                cv::Vec3b col1 = img1.at<cv::Vec3b>(y, x);
#if ENABLE_MY_MATCHING
                cv::Point2f pt = phg::transformPoint(cv::Point2f(x, y), H);
#else
                cv::Point2f pt = phg::transformPointCV(cv::Point2f(x, y), H);
#endif
                int pt_x = std::round(pt.x);
                int pt_y = std::round(pt.y);
                if (pt_x >= 0 && pt_x < img2.cols && pt_y >= 0 && pt_y < img2.rows) {
                    cv::Vec3b col2 = img2.at<cv::Vec3b>(pt_y, pt_x);
                    int dc0 = int(col2[0]) - int(col1[0]);
                    int dc1 = int(col2[1]) - int(col1[1]);
                    int dc2 = int(col2[2]) - int(col1[2]);

                    color_rmse += dc0 * dc0 + dc1 * dc1 + dc2 * dc2;
                    ++count;
                }
            }
        }

        if (count) {
            color_rmse /= count;
            color_rmse = std::sqrt(color_rmse);
        }
    }

}

namespace {
    void testStitching(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                       const cv::Mat &descriptors1, const cv::Mat &descriptors2)
    {
        double rmse_kpts, rmse_color;
        evaluateStitching(img1, img2, rmse_kpts, rmse_color, keypoints1, keypoints2, descriptors1, descriptors2);

        std::cout << "keypoints RMSE: " << rmse_kpts << ", color RMSE: " << rmse_color << std::endl;

        EXPECT_LT(rmse_kpts, max_keypoints_rmse_px);
        EXPECT_LT(rmse_color, max_color_rmse_8u);
    }

    void testStitchingMultipleDetectors(const cv::Mat &img1, const cv::Mat &img2)
    {
        {
            std::cout << "testing sift detector/descriptor..." << std::endl;
            cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            cv::Mat descriptors1, descriptors2;
            detector->detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1 );
            detector->detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2 );

            testStitching(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2);
        }

#if ENABLE_MY_DESCRIPTOR
        {
            std::cout << "testing my detector/descriptor..." << std::endl;
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            cv::Mat descriptors1, descriptors2;
            phg::SIFT mySIFT;
            mySIFT.detectAndCompute(img1, keypoints1, descriptors1);
            mySIFT.detectAndCompute(img2, keypoints2, descriptors2);

            testStitching(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2);
        }
#endif
    }
}

TEST (MATCHING, SimpleStitching) {

    cv::Mat img1 = cv::imread("data/src/test_matching/hiking_left.JPG");
    cv::Mat img2 = cv::imread("data/src/test_matching/hiking_right.JPG");

    testStitchingMultipleDetectors(img1, img2);
}

namespace {

    bool matcheq(const cv::DMatch &lhs, const cv::DMatch &rhs)
    {
        return std::tie(lhs.trainIdx, lhs.queryIdx, lhs.imgIdx) == std::tie(rhs.trainIdx, rhs.queryIdx, rhs.imgIdx);
    }

    void evaluateMatching(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                          const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                          double &nn_score, double &nn2_score, double &nn_score_cv, double &nn2_score_cv,
                          double &time_my, double &time_cv, double &time_bruteforce, double &time_bruteforce_gpu,
                          double &good_nn, double &good_ratio, double &good_clusters, double &good_ratio_and_clusters, bool do_bruteforce)
    {
        using namespace cv;

        if (!descriptors1.rows || !descriptors2.rows) {
            throw std::runtime_error("empty descriptors");
        }

        timer tm;

        std::vector<std::vector<DMatch>> knn_matches_flann, knn_matches_flann_cv, knn_matches_bruteforce;

        std::cout << "flann matching..." << std::endl;
        tm.restart();
        #if ENABLE_MY_MATCHING
        {
            phg::FlannMatcher matcher;
            matcher.train(descriptors2);
            matcher.knnMatch(descriptors1, knn_matches_flann, 2);
        }
        #endif
        time_my = tm.elapsed();

        std::cout << "cv flann matching..." << std::endl;
        tm.restart();
        {
            Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
            matcher->knnMatch( descriptors1, descriptors2, knn_matches_flann_cv, 2 );

            #if !ENABLE_MY_MATCHING
            knn_matches_flann = knn_matches_flann_cv;
            #endif
        }
        time_cv = tm.elapsed();

        tm.restart();
        if (do_bruteforce) {
            std::cout << "brute force matching" << std::endl;
            phg::BruteforceMatcher matcher;
            matcher.train(descriptors2);
            matcher.knnMatch(descriptors1, knn_matches_bruteforce, 2);
        }
        time_bruteforce = tm.elapsed();

        tm.restart();
        std::vector<std::vector<DMatch>> knn_matches_bruteforce_gpu;
#if ENABLE_GPU_BRUTEFORCE_MATCHER
        if (do_bruteforce) {
            std::cout << "brute force GPU matching" << std::endl;
            phg::BruteforceMatcherGPU matcher;
            matcher.train(descriptors2);
            matcher.knnMatch(descriptors1, knn_matches_bruteforce_gpu, 2);
        }
#endif
        time_bruteforce_gpu = tm.elapsed();

#if ENABLE_GPU_BRUTEFORCE_MATCHER
        ASSERT_EQ(knn_matches_bruteforce_gpu.size(), knn_matches_bruteforce.size());
        for (int i = 0; i < (int) knn_matches_bruteforce_gpu.size(); ++i) {
            ASSERT_EQ(knn_matches_bruteforce_gpu[i].size(), knn_matches_bruteforce[i].size());
            for (int j = 0; j < (int) knn_matches_bruteforce_gpu[i].size(); ++j) {
                ASSERT_TRUE(matcheq(knn_matches_bruteforce_gpu[i][j], knn_matches_bruteforce[i][j]));
            }
        }
#endif

        nn_score = 0;
        nn2_score = 0;
        nn_score_cv = 0;
        nn2_score_cv = 0;
        if (do_bruteforce) {
            for (int i = 0; i < descriptors1.rows; ++i) {
                if (knn_matches_bruteforce[i][0].queryIdx != i) {
                    throw std::runtime_error("invalid DMatch queryIdx for knn_matches_bruteforce");
                }
                if (knn_matches_flann[i][0].queryIdx != i) {
                    throw std::runtime_error("invalid DMatch queryIdx for knn_matches_flann");
                }
                if (knn_matches_flann_cv[i][0].queryIdx != i) {
                    throw std::runtime_error("invalid DMatch queryIdx for knn_matches_flann_cv");
                }

                if (knn_matches_flann[i][0].trainIdx == knn_matches_bruteforce[i][0].trainIdx) {
                    ++nn_score;
                }

                if (knn_matches_flann[i][1].trainIdx == knn_matches_bruteforce[i][1].trainIdx) {
                    ++nn2_score;
                }

                if (knn_matches_flann_cv[i][0].trainIdx == knn_matches_bruteforce[i][0].trainIdx) {
                    ++nn_score_cv;
                }

                if (knn_matches_flann_cv[i][1].trainIdx == knn_matches_bruteforce[i][1].trainIdx) {
                    ++nn2_score_cv;
                }
            }

            nn_score /= descriptors1.rows;
            nn2_score /= descriptors1.rows;
            nn_score_cv /= descriptors1.rows;
            nn2_score_cv /= descriptors1.rows;
        }

        std::vector<DMatch> good_matches_nn(knn_matches_flann.size());
        for (int i = 0; i < (int) knn_matches_flann.size(); ++i) {
            good_matches_nn[i] = knn_matches_flann[i][0];
        }
        drawMatches(img1, img2, keypoints1, keypoints2, good_matches_nn, "data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "00_matches_nn.png");

        #if ENABLE_MY_MATCHING
        std::cout << "filtering matches by ratio test..." << std::endl;
        std::vector<DMatch> good_matches_ratio;
        phg::DescriptorMatcher::filterMatchesRatioTest(knn_matches_flann, good_matches_ratio);
        drawMatches(img1, img2, keypoints1, keypoints2, good_matches_ratio, "data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "01_matches_ratio.png");

        std::cout << "filtering matches by clusters..." << std::endl;
        std::vector<DMatch> good_matches_clusters_only;
        phg::DescriptorMatcher::filterMatchesClusters(good_matches_nn, keypoints1, keypoints2, good_matches_clusters_only);
        drawMatches(img1, img2, keypoints1, keypoints2, good_matches_clusters_only, "data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "03_matches_clusters_only.png");

        std::cout << "filtering matches by ratio & clusters" << std::endl;
        std::vector<DMatch> good_matches_clusters_and_ratio;
        phg::DescriptorMatcher::filterMatchesClusters(good_matches_ratio, keypoints1, keypoints2, good_matches_clusters_and_ratio);
        drawMatches(img1, img2, keypoints1, keypoints2, good_matches_clusters_and_ratio, "data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "04_matches_clusters_and_ratio.png");
        #else
        std::vector<DMatch> good_matches_clusters_and_ratio;
        phg::filterMatchesGMS(good_matches_nn, keypoints1, keypoints2, img1.size(), img2.size(), good_matches_clusters_and_ratio);
        drawMatches(img1, img2, keypoints1, keypoints2, good_matches_clusters_and_ratio, "data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "04_matches_gms.png");
        #endif

        std::cout << "estimating homography..." << std::endl;
        cv::Mat H;
        {
            std::vector<cv::Point2f> points1, points2;
            for (const cv::DMatch &match : good_matches_clusters_and_ratio) {
                points1.push_back(keypoints1[match.queryIdx].pt);
                points2.push_back(keypoints2[match.trainIdx].pt);
            }

#if ENABLE_MY_MATCHING
            H = phg::findHomography(points1, points2);
#else
            H = phg::findHomographyCV(points1, points2);
#endif
        }

        good_nn = 0;
        good_ratio = 0;
        good_clusters = 0;
        good_ratio_and_clusters = 0;

        std::cout << "evaluating homography..." << std::endl;

        #if ENABLE_MY_MATCHING
        const int ntest = 4;
        std::vector<cv::DMatch>* arrs[ntest] = {&good_matches_nn, &good_matches_ratio, &good_matches_clusters_only, &good_matches_clusters_and_ratio};
        double* ptrs[ntest] = {&good_nn, &good_ratio, &good_clusters, &good_ratio_and_clusters};
        #else
        const int ntest = 2;
        std::vector<cv::DMatch>* arrs[ntest] = {&good_matches_nn, &good_matches_clusters_and_ratio};
        double* ptrs[ntest] = {&good_nn, &good_ratio_and_clusters};
        #endif

        for (int i_test = 0; i_test < ntest; ++i_test) {

            const std::vector<cv::DMatch> &arr = *arrs[i_test];

            if (arr.size() < 50) {
                std::cerr << "too few matches: " + std::to_string(arr.size()) << std::endl;
                continue;
            }

            std::vector<cv::Point2f> points1, points2;
            for (const cv::DMatch &match : arr) {
                points1.push_back(keypoints1[match.queryIdx].pt);
                points2.push_back(keypoints2[match.trainIdx].pt);
            }

            (*ptrs[i_test]) = 0;
            for (int i = 0; i < (int) arr.size(); ++i) {
#if ENABLE_MY_MATCHING
                cv::Point2f pt = phg::transformPoint(points1[i], H);
#else
                cv::Point2f pt = phg::transformPointCV(points1[i], H);
#endif

                cv::Point2f diff = pt - points2[i];
                float dist2 = diff.x * diff.x + diff.y * diff.y;
                if (dist2 < max_keypoints_rmse_px * max_keypoints_rmse_px) {
                    ++(*ptrs[i_test]);
                }
            }
            if (arr.size()) {
                (*ptrs[i_test]) /= arr.size();
            }
        }

    }

    void testMatching(const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                       const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                      double &nn_score, double &nn2_score, double &nn_score_cv, double &nn2_score_cv,
                      double &time_my, double &time_cv, double &time_bruteforce, double &time_bruteforce_gpu,
                      double &good_nn, double &good_ratio, double &good_clusters, double &good_ratio_and_clusters,
                      bool do_bruteforce
                       )
    {
        evaluateMatching(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2,
                         nn_score, nn2_score, nn_score_cv, nn2_score_cv,
                         time_my, time_cv, time_bruteforce, time_bruteforce_gpu,
                         good_nn, good_ratio, good_clusters, good_ratio_and_clusters, do_bruteforce);

        std::cout << "nn_score: " << nn_score << ", ";
        std::cout << "nn2_score: " << nn2_score << ", ";
        std::cout << "nn_score_cv: " << nn_score_cv << ", ";
        std::cout << "nn2_score_cv: " << nn2_score_cv << ", ";
        std::cout << "time_my: " << time_my << ", ";
        std::cout << "time_cv: " << time_cv << ", ";
        std::cout << "time_bruteforce: " << time_bruteforce << ", ";
#if ENABLE_GPU_BRUTEFORCE_MATCHER
        std::cout << "time_bruteforce_gpu: " << time_bruteforce_gpu << ", ";
#endif
        std::cout << "good_nn: " << good_nn << ", ";
        std::cout << "good_ratio: " << good_ratio << ", ";
        std::cout << "good_clusters: " << good_clusters << ", ";
        std::cout << "good_ratio_and_clusters: " << good_ratio_and_clusters << std::endl;
    }

    void testMatchingMultipleDetectors(const cv::Mat &img1, const cv::Mat &img2,
                                        double &nn_score, double &nn2_score, double &nn_score_cv, double &nn2_score_cv,
                                        double &time_my, double &time_cv, double &time_bruteforce, double &time_bruteforce_gpu,
                                        double &good_nn, double &good_ratio, double &good_clusters, double &good_ratio_and_clusters, bool do_bruteforce = true)
    {
        {
            std::cout << "testing sift detector/descriptor..." << std::endl;
            cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            cv::Mat descriptors1, descriptors2;
            detector->detectAndCompute( img1, cv::noArray(), keypoints1, descriptors1 );
            detector->detectAndCompute( img2, cv::noArray(), keypoints2, descriptors2 );

            testMatching(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2,
                         nn_score, nn2_score, nn_score_cv, nn2_score_cv,
                         time_my, time_cv, time_bruteforce, time_bruteforce_gpu,
                         good_nn, good_ratio, good_clusters, good_ratio_and_clusters, do_bruteforce);
        }
#if ENABLE_MY_DESCRIPTOR
        {
            std::cout << "testing my detector/descriptor..." << std::endl;
            std::vector<cv::KeyPoint> keypoints1, keypoints2;
            cv::Mat descriptors1, descriptors2;
            phg::SIFT mySIFT;
            mySIFT.detectAndCompute(img1, keypoints1, descriptors1);
            mySIFT.detectAndCompute(img2, keypoints2, descriptors2);

            testMatching(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2,
                         nn_score, nn2_score, nn_score_cv, nn2_score_cv,
                         time_my, time_cv, time_bruteforce, time_bruteforce_gpu,
                         good_nn, good_ratio, good_clusters, good_ratio_and_clusters, do_bruteforce);
        }
#endif
    }

}

TEST (MATCHING, SimpleMatching) {

    cv::Mat img1 = cv::imread("data/src/test_matching/hiking_left.JPG");
    cv::Mat img2 = cv::imread("data/src/test_matching/hiking_right.JPG");


    double nn_score, nn2_score, nn_score_cv, nn2_score_cv,
            time_my, time_cv, time_bruteforce, time_bruteforce_gpu, good_nn, good_ratio, good_clusters, good_ratio_and_clusters;

    testMatchingMultipleDetectors(img1, img2,
                                  nn_score, nn2_score, nn_score_cv, nn2_score_cv,
                                  time_my, time_cv, time_bruteforce, time_bruteforce_gpu,
                                  good_nn, good_ratio, good_clusters, good_ratio_and_clusters);



    EXPECT_GT(nn_score, 0.9 * nn_score_cv);
    EXPECT_GT(nn2_score, 0.9 * nn2_score_cv);

    EXPECT_LT(time_my, 1.5 * time_cv);
    EXPECT_LT(time_my, 0.1 * time_bruteforce);

#if ENABLE_GPU_BRUTEFORCE_MATCHER
    EXPECT_LT(time_bruteforce_gpu, time_bruteforce);
#endif

#if ENABLE_MY_MATCHING
    EXPECT_LT(good_nn, good_ratio);
    EXPECT_LT(good_nn, good_clusters);
#endif
    EXPECT_LT(good_nn, good_ratio_and_clusters);

    EXPECT_GT(good_nn, 0.2);
#if ENABLE_MY_MATCHING
    EXPECT_GT(good_ratio, 0.9);
    EXPECT_GT(good_clusters, 0.9);
#endif
    EXPECT_GT(good_ratio_and_clusters, 0.9);
}

namespace {

    cv::Mat transformImg(const cv::Mat &img2, double angleDegreesClockwise, double scale)
    {
        cv::Mat M = cv::getRotationMatrix2D(cv::Point(0, 0), 0, scale);
        M.at<double>(0, 2) = img2.cols * 0.25 * scale;
        M.at<double>(1, 2) = img2.rows * 0.25 * scale;
        cv::Mat tmp;
        cv::warpAffine(img2, tmp, M, cv::Size(1.5 * img2.cols * scale, 1.5 * img2.rows * scale));

        cv::Mat transformedImage;
        M = cv::getRotationMatrix2D(cv::Point(tmp.cols / 2, tmp.rows / 2), -angleDegreesClockwise, 1.0);
        cv::warpAffine(tmp, transformedImage, M, cv::Size(tmp.cols, tmp.rows));

        return transformedImage;
    }

    void addNoise(cv::Mat &img2)
    {
        cv::Mat noise(cv::Size(img2.cols, img2.rows), CV_8UC3);
        cv::setRNGSeed(125125); // фиксируем рандом для детерминизма (чтобы результат воспроизводился из раза в раз)
        cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(GAUSSIAN_NOISE_STDDEV));
        cv::add(img2, noise, img2); // добавляем к преобразованной картинке гауссиан шума
    }

    void testMatchingTransformWrapper(double angleDegreesClockwise, double scale)
    {
        cv::Mat img1 = cv::imread("data/src/test_matching/hiking_left.JPG");
        cv::Mat img2 = cv::imread("data/src/test_matching/hiking_right.JPG");

        img2 = transformImg(img2, angleDegreesClockwise, scale);
        addNoise(img2);

        cv::imwrite("data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "hiking_right_rotated_noise.png", img2);

        double nn_score, nn2_score, nn_score_cv, nn2_score_cv,
                time_my, time_cv, time_bruteforce, time_bruteforce_gpu, good_nn, good_ratio, good_clusters, good_ratio_and_clusters;

        testMatchingMultipleDetectors(img1, img2,
                                      nn_score, nn2_score, nn_score_cv, nn2_score_cv,
                                      time_my, time_cv, time_bruteforce, time_bruteforce_gpu,
                                      good_nn, good_ratio, good_clusters, good_ratio_and_clusters, false);

        EXPECT_LT(time_my, 1.5 * time_cv);

#if ENABLE_MY_MATCHING
        EXPECT_LT(good_nn, good_ratio);
#endif
        EXPECT_LT(good_nn, good_ratio_and_clusters);

#if ENABLE_MY_MATCHING
        EXPECT_GT(good_ratio, 0.7);
#endif
        EXPECT_GT(good_ratio_and_clusters, 0.7);
    }

}

TEST (MATCHING, Rotate10) {
    double angleDegreesClockwise = 10;
    double scale = 1.0;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Rotate20) {
    double angleDegreesClockwise = 20;
    double scale = 1.0;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Rotate30) {
    double angleDegreesClockwise = 30;
    double scale = 1.0;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Rotate40) {
    double angleDegreesClockwise = 40;
    double scale = 1.0;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Rotate45) {
    double angleDegreesClockwise = 45;
    double scale = 1.0;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Rotate90) {
    double angleDegreesClockwise = 90;
    double scale = 1.0;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Scale50) {
    // seems to be some issue with gms matcher and high downscale
#if ENABLE_MY_MATCHING
    double angleDegreesClockwise = 0;
    double scale = 0.5;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
#endif
}

TEST (MATCHING, Scale70) {
    double angleDegreesClockwise = 0;
    double scale = 0.7;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Scale90) {
    double angleDegreesClockwise = 0;
    double scale = 0.9;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Scale110) {
    double angleDegreesClockwise = 0;
    double scale = 1.1;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Scale130) {
    double angleDegreesClockwise = 0;
    double scale = 1.3;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Scale150) {
    double angleDegreesClockwise = 0;
    double scale = 1.5;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Scale175) {
    double angleDegreesClockwise = 0;
    double scale = 1.75;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Scale200) {
    double angleDegreesClockwise = 0;
    double scale = 2.0;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Rotate10Scale90) {
    double angleDegreesClockwise = 10;
    double scale = 0.9;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (MATCHING, Rotate30Scale75) {
    double angleDegreesClockwise = 30;
    double scale = 0.75;

    testMatchingTransformWrapper(angleDegreesClockwise, scale);
}

TEST (STITCHING, SimplePanorama) {
#if ENABLE_MY_MATCHING
    cv::Mat img1 = cv::imread("data/src/test_matching/hiking_left.JPG");
    cv::Mat img2 = cv::imread("data/src/test_matching/hiking_right.JPG");

    std::function<cv::Mat(const cv::Mat&, const cv::Mat&)> homography_builder = [](const cv::Mat &lhs, const cv::Mat &rhs){ return getHomography(lhs, rhs); };
    cv::Mat pano = phg::stitchPanorama({img1, img2}, {-1, 0}, homography_builder);
    cv::imwrite("data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "panorama.png", pano);
#endif
}

namespace {

    int getOrthoScore(const cv::Mat &ortho0, const cv::Mat &ortho1, int threshold_px)
    {
        using namespace cv;

        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        detector->detectAndCompute( ortho0, cv::noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( ortho1, cv::noArray(), keypoints2, descriptors2 );

        std::vector< std::vector<DMatch> > knn_matches;

        phg::FlannMatcher matcher;
        matcher.train(descriptors2);
        matcher.knnMatch(descriptors1, knn_matches, 2);

        std::vector<DMatch> good_matches(knn_matches.size());
        for (int i = 0; i < (int) knn_matches.size(); ++i) {
            good_matches[i] = knn_matches[i][0];
        }

        phg::DescriptorMatcher::filterMatchesRatioTest(knn_matches, good_matches);

        {
            std::vector<DMatch> tmp;
            phg::DescriptorMatcher::filterMatchesClusters(good_matches, keypoints1, keypoints2, tmp);
            std::swap(tmp, good_matches);
        }

        int score = 0;
        for (const cv::DMatch &match : good_matches) {
            cv::Point2f d = keypoints1[match.queryIdx].pt - keypoints2[match.trainIdx].pt;
            if (d.x * d.x + d.y * d.y < threshold_px * threshold_px) {
                ++score;
            }
        }

        drawMatches(ortho0, ortho1, keypoints1, keypoints2, good_matches, "data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "ortho_matches.png");

        return score;
    }

}

TEST (STITCHING, Orthophoto) {
#if ENABLE_MY_MATCHING
    cv::Mat img1 = cv::imread("data/src/test_matching/ortho/IMG_160729_071349_0000_RGB.JPG");
    cv::Mat img2 = cv::imread("data/src/test_matching/ortho/IMG_160729_071351_0001_RGB.JPG");
    cv::Mat img3 = cv::imread("data/src/test_matching/ortho/IMG_160729_071353_0002_RGB.JPG");
    cv::Mat img4 = cv::imread("data/src/test_matching/ortho/IMG_160729_071356_0003_RGB.JPG");
    cv::Mat img5 = cv::imread("data/src/test_matching/ortho/IMG_160729_071358_0004_RGB.JPG");

    {
        std::function<cv::Mat(const cv::Mat&, const cv::Mat&)> homography_builder = [](const cv::Mat &lhs, const cv::Mat &rhs){ return getHomography(lhs, rhs); };
        cv::Mat ortho2 = phg::stitchPanorama({img1, img2, img3, img4, img5}, {1, 2, -1, 2, 3}, homography_builder);
        cv::imwrite("data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "ortho_root2.jpg", ortho2);
    }

    int counter = 0;
    std::function<cv::Mat(const cv::Mat&, const cv::Mat&)> homography_builder = [&counter](const cv::Mat &lhs, const cv::Mat &rhs){
        ++counter;
        return getHomography(lhs, rhs);
    };

    cv::Mat ortho = phg::stitchPanorama({img1, img2, img3, img4, img5}, {-1, 0, 1, 2, 3}, homography_builder);
    cv::imwrite("data/debug/test_matching/" + getTestSuiteName() + "_" + getTestName() + "_" + "ortho_root0.jpg", ortho);

    // гомография должна быть посчитана для каждого ребра в графе по разу
    EXPECT_EQ(counter, 4);

    int threshold_px = 250;
    int score = getOrthoScore(ortho, cv::imread("data/src/test_matching/ortho/ortho_root0.jpg"), threshold_px);
    std::cout << "n stable ortho kpts: : " << score << std::endl;
    EXPECT_GT(score, 7500);
#endif
}