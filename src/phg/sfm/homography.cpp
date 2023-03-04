#include "homography.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

namespace {

    // источник: https://e-maxx.ru/algo/linear_systems_gauss
    // очень важно при выполнении метода гаусса использовать выбор опорного элемента: об этом можно почитать в источнике кода
    // или на вики: https://en.wikipedia.org/wiki/Pivot_element
    int gauss(std::vector<std::vector<double>> a, std::vector<double> &ans)
    {
        using namespace std;
        const double EPS = 1e-8;
        const int INF = std::numeric_limits<int>::max();

        int n = (int) a.size();
        int m = (int) a[0].size() - 1;

        vector<int> where (m, -1);
        for (int col=0, row=0; col<m && row<n; ++col) {
            int sel = row;
            for (int i=row; i<n; ++i)
                if (abs (a[i][col]) > abs (a[sel][col]))
                    sel = i;
            if (abs (a[sel][col]) < EPS)
                continue;
            for (int i=col; i<=m; ++i)
                swap (a[sel][i], a[row][i]);
            where[col] = row;

            for (int i=0; i<n; ++i)
                if (i != row) {
                    double c = a[i][col] / a[row][col];
                    for (int j=col; j<=m; ++j)
                        a[i][j] -= a[row][j] * c;
                }
            ++row;
        }

        ans.assign (m, 0);
        for (int i=0; i<m; ++i)
            if (where[i] != -1)
                ans[i] = a[where[i]][m] / a[where[i]][i];
        for (int i=0; i<n; ++i) {
            double sum = 0;
            for (int j=0; j<m; ++j)
                sum += ans[j] * a[i][j];
            if (abs (sum - a[i][m]) > EPS)
                return 0;
        }

        for (int i=0; i<m; ++i)
            if (where[i] == -1)
                return INF;
        return 1;
    }

    // см. Hartley, Zisserman: Multiple View Geometry in Computer Vision. Second Edition 4.1, 4.1.2
    cv::Mat estimateHomography4Points(const cv::Point2f &l0, const cv::Point2f &l1,
                                      const cv::Point2f &l2, const cv::Point2f &l3,
                                      const cv::Point2f &r0, const cv::Point2f &r1,
                                      const cv::Point2f &r2, const cv::Point2f &r3)
    {
        std::vector<std::vector<double>> A;
        std::vector<double> H;

        double xs0[4] = {l0.x, l1.x, l2.x, l3.x};
        double xs1[4] = {r0.x, r1.x, r2.x, r3.x};
        double ys0[4] = {l0.y, l1.y, l2.y, l3.y};
        double ys1[4] = {r0.y, r1.y, r2.y, r3.y};
        double ws0[4] = {1, 1, 1, 1};
        double ws1[4] = {1, 1, 1, 1};

        for (int i = 0; i < 4; ++i) {
            // fill 2 rows of matrix A

            double x0 = xs0[i];
            double y0 = ys0[i];
            double w0 = ws0[i];

            double x1 = xs1[i];
            double y1 = ys1[i];
            double w1 = ws1[i];

            // 8 elements of matrix + free term as needed by gauss routine
            A.push_back({
                    0.0, 0.0, 0.0,
                    -w1*x0, -w1*y0, -w1*w0,
                    y1*x0, y1*y0, -y1*w0
            });
            A.push_back({
                    w1*x0, w1*y0, w1*w0,
                    0.0, 0.0, 0.0,
                    -x1*x0, -x1*y0, x1*w0
            });
        }

        int res = gauss(A, H);
        if (res == 0) {
            throw std::runtime_error("gauss: no solution found");
        }
        else
        if (res == 1) {
            //std::cout << "gauss: unique solution found" << std::endl;
        }
        else
        if (res == std::numeric_limits<int>::max()) {
            std::cerr << "gauss: infinitely many solutions found" << std::endl;
            std::cerr << "gauss: xs0: ";
            for (int i = 0; i < 4; ++i) {
                std::cerr << xs0[i] << ", ";
            }
            std::cerr << "\ngauss: ys0: ";
            for (int i = 0; i < 4; ++i) {
                std::cerr << ys0[i] << ", ";
            }
            std::cerr << std::endl;
        }
        else
        {
            throw std::runtime_error("gauss: unexpected return code");
        }

        // add fixed element H33 = 1
        H.push_back(1.0);

        cv::Mat H_mat(3, 3, CV_64FC1);
        std::copy(H.begin(), H.end(), H_mat.ptr<double>());
        return H_mat;
    }

    // pseudorandom number generator
    inline uint64_t xorshift64(uint64_t *state)
    {
        if (*state == 0) {
            *state = 1;
        }

        uint64_t x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        return *state = x;
    }

    void randomSample(std::vector<int> &dst, int max_id, int sample_size, uint64_t *state)
    {
        dst.clear();

        const int max_attempts = 1000;

        for (int i = 0; i < sample_size; ++i) {
            for (int k = 0; k < max_attempts; ++k) {
                int v = xorshift64(state) % max_id;
                if (dst.empty() || std::find(dst.begin(), dst.end(), v) == dst.end()) {
                    dst.push_back(v);
                    break;
                }
            }
            if (dst.size() < i + 1) {
                throw std::runtime_error("Failed to sample ids");
            }
        }
    }

    cv::Mat estimateHomographyRANSAC(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
    {
        if (points_lhs.size() != points_rhs.size()) {
            throw std::runtime_error("findHomography: points_lhs.size() != points_rhs.size()");
        }

        // TODO Дополнительный балл, если вместо обычной версии будет использована модификация a-contrario RANSAC
        // * [1] Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers. (Lionel Moisan, Pierre Moulon, Pascal Monasse)
        // * [2] Adaptive Structure from Motion with a contrario model estimation. (Pierre Moulon, Pascal Monasse, Renaud Marlet)
        // * (простое описание для понимания)
        // * [3] http://ikrisoft.blogspot.com/2015/01/ransac-with-contrario-approach.html

        const int n_matches = points_lhs.size();

        // https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
        const int n_trials = 50;

        const int n_samples = 4;
        uint64_t seed = 1;
        const double reprojection_error_threshold_px = 2;

        int best_support = 0;
        cv::Mat best_H;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            randomSample(sample, n_matches, n_samples, &seed);

            cv::Mat H = estimateHomography4Points(
                    points_lhs[sample[0]], points_lhs[sample[1]],
                    points_lhs[sample[2]], points_lhs[sample[3]],
                    points_rhs[sample[0]], points_rhs[sample[1]],
                    points_rhs[sample[2]], points_rhs[sample[3]]
            );

            int support = 0;
            for (int i_point = 0; i_point < n_matches; ++i_point) {
                try {
                    cv::Point2d proj = phg::transformPoint(points_lhs[i_point], H);
                    if (cv::norm(proj - cv::Point2d(points_rhs[i_point])) < reprojection_error_threshold_px) {
                        ++support;
                    }
                } catch (const std::exception &e)
                {
                    std::cerr << e.what() << std::endl;
                }
            }

            if (support > best_support) {
                best_support = support;
                best_H = H;

                std::cout << "estimateHomographyRANSAC : support: " << best_support << "/" << n_matches << std::endl;

                if (best_support == n_matches) {
                    break;
                }
            }
        }

        std::cout << "estimateHomographyRANSAC : best support: " << best_support << "/" << n_matches << std::endl;

        if (best_support == 0) {
            throw std::runtime_error("estimateHomographyRANSAC : failed to estimate homography");
        }

        return best_H;
    }

}

cv::Mat phg::findHomography(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
{
    return estimateHomographyRANSAC(points_lhs, points_rhs);
}

// чтобы заработало, нужно пересобрать библиотеку с дополнительным модулем calib3d (см. инструкцию в корневом CMakeLists.txt)
cv::Mat phg::findHomographyCV(const std::vector<cv::Point2f> &points_lhs, const std::vector<cv::Point2f> &points_rhs)
{
    return cv::findHomography(points_lhs, points_rhs, cv::RANSAC);
}

// T - 3x3 однородная матрица, например, гомография
// таким преобразованием внутри занимается функции cv::perspectiveTransform и cv::warpPerspective
cv::Point2d phg::transformPoint(const cv::Point2d &pt, const cv::Mat &T)
{
    double data[3][1] = {{pt.x}, {pt.y}, {1.0}};
    cv::Mat transformed = T * cv::Mat(3, 1, CV_64F, data);
    double z = transformed.at<double>(2, 0);
    return cv::Point2d(transformed.at<double>(0,0)/z, transformed.at<double>(1,0)/z);
}

cv::Point2d phg::transformPointCV(const cv::Point2d &pt, const cv::Mat &T) {
    // ineffective but ok for testing
    std::vector<cv::Point2f> tmp0 = {pt};
    std::vector<cv::Point2f> tmp1(1);
    cv::perspectiveTransform(tmp0, tmp1, T);
    return tmp1[0];
}
