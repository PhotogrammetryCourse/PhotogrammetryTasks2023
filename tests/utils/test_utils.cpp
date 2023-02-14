#include "test_utils.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <libutils/rasserts.h>


cv::Mat concatenateImagesLeftRight(const cv::Mat &img0, const cv::Mat &img1) {
    // это способ гарантировать себе что предположение которое явно в этой функции есть (совпадение типов картинок)
    // однажды не нарушится (по мере изменения кода) и не приведет к непредсказуемым последствиям
    // в отличие от assert() у таких rassert есть три преимущества:
    // 1) они попадают в т.ч. в релизную сборку
    // 2) есть (псевдо)уникальный идентификатор по которому легко найти где это произошло
    //    (в отличие от просто __LINE__, т.к. даже если исходный файл угадать и легко, то нумерация строк может меняться от коммита к коммиту,
    //     а падение могло случится у пользователя на старорй версии)
    // 3) есть общая удобная точка остановки на которую легко поставить breakpoint - rasserts.cpp/debugPoint()
    rassert(img0.type() == img1.type(), 125121612363131);
    rassert(img0.channels() == img1.channels(), 136161251414);

    size_t width = img0.cols + img1.cols;
    size_t height = std::max(img0.rows, img1.rows);

    cv::Mat res(height, width, img0.type());
    img0.copyTo(res(cv::Rect(0, 0, img0.cols, img0.rows)));
    img1.copyTo(res(cv::Rect(img0.cols, 0, img1.cols, img1.rows)));

    return res;
}


std::string getTestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}


std::string getTestSuiteName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
}
