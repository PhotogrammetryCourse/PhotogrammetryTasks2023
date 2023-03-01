#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <libutils/timer.h>
#include <libutils/rasserts.h>

#include <phg/sift/sift.h>

#include "utils/test_utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define SHOW_RESULTS                0   // если вам хочется сразу видеть результат в окошке - переключите в 1, но не забудьте выключить перед коммитом (иначе бот в CI будет ждать веками)
#define MAX_ACCEPTED_PIXEL_ERROR    0.01 // максимальное расстояние в пикселях (процент от ширины картинки) между ключевыми точками чтобы их можно было зачесть как "почти совпавшие" (это очень завышенный порог, по-хорошему должно быть 0.5 например)
#define MAX_AVG_PIXEL_ERROR         0.075

#define GAUSSIAN_NOISE_STDDEV       1.0

//#define imread(path)           imread("../" + std::string(path))


// функция рисует кружки случайного цвета вокруг точек, но если для точки не нашлось сопоставления - кружок будет толстый и ярко красный
void drawKeyPoints(cv::Mat &img, const std::vector<cv::KeyPoint> &kps, const std::vector<unsigned char> &is_not_matched) {
    cv::RNG r(124124);
    for (size_t i = 0; i < kps.size(); ++i) {
        int thickness = 1;
        cv::Scalar color;
        if (is_not_matched[i]) {
            color = CV_RGB(255, 0, 0); // OpenCV использует BGR схему вместо RGB, но можно использовать этот макрос вместо BGR - cv::Scalar(blue=0, green=0, red=255)  
            thickness = 2;
        } else {
            color = cv::Scalar(r.uniform(0, 255), r.uniform(0, 255), 0);
        }
        int radius = std::max(2, (int) (kps[i].size / 5.0f));
        float angle = kps[i].angle;
        cv::circle(img, kps[i].pt, radius, color, thickness);
        if (angle != -1.0) {
            cv::line(img, kps[i].pt, cv::Point((int) std::round(kps[i].pt.x + radius*sin(angle/M_PI)), (int) std::round(kps[i].pt.y + radius*cos(angle/M_PI))), color);
        }
    }
}

// Функция ищет знаковый угол между двумя направлениями (по кратчайшему пути, т.е. результат от -180 до 180)
double diffAngles(double angle0, double angle1) {
    if (angle0 != -1.0 && angle1 != -1.0) {
        rassert(angle0 >= 0.0 && angle0 < 360.0, 1235612352151);
        rassert(angle1 >= 0.0 && angle1 < 360.0, 4645315415);
        float diff;
        if ((angle1 <= angle0 + 180 && angle0 + 180 <= 360) || (angle1 >= angle0 - 180 && angle0 - 180 >= 0)) {
            diff = angle1 - angle0;
        } else if (angle1 > angle0 + 180 && angle0 + 180 <= 360) {
            diff = -(angle0 + (360 - angle1));
        } else if (angle1 <= angle0 - 180 && angle0 - 180 >= 0) {
            diff = (360 - angle0) + angle1;
        } else {
            rassert(false, 1234124125125135);
        }
        rassert(diff >= -180 && diff <= 180, 233536136131);
        return diff;
    } else {
        return 0.0;
    }
}

// На вход передается матрица описывающая преобразование картинки (сдвиг, поворот, масштабирование или их комбинация), допустимый процент Recall, и опционально можно тестировать другую картинку
void evaluateDetection(const cv::Mat &M, double minRecall, cv::Mat img0=cv::Mat()) {
    if (img0.empty()) {
        img0 = cv::imread("data/src/test_sift/unicorn.png"); // грузим картинку по умолчанию
    }

    ASSERT_FALSE(img0.empty()); // проверка что картинка была загружена
    // убедитесь что рабочая папка (Edit Configurations...->Working directory) указывает на корневую папку проекта (и тогда картинка по умолчанию найдется по относительному пути - data/src/test_sift/unicorn.png)
    
    size_t width = img0.cols;
    size_t height = img0.rows;
    cv::Mat transformedImage;
    cv::warpAffine(img0, transformedImage, M, cv::Size(width, height)); // строим img1 - преобразованная исходная картинка в соответствии с закодированным в матрицу M искажением пространства
    cv::Mat noise(cv::Size(width, height), CV_8UC3);
    cv::setRNGSeed(125125); // фиксируем рандом для детерминизма (чтобы результат воспроизводился из раза в раз)
    cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(GAUSSIAN_NOISE_STDDEV));
    cv::add(transformedImage, noise, transformedImage); // добавляем к преобразованной картинке гауссиан шума
    cv::Mat img1 = transformedImage;

    {
        for (int method = 0; method < 3; ++method) { // тестируем три метода: OpenCV ORB, OpenCV SIFT, ваш SIFT
            std::vector<cv::KeyPoint> kps0;
            std::vector<cv::KeyPoint> kps1;

            cv::Mat desc0;
            cv::Mat desc1;

            timer t; // очень удобно встраивать профилирование вашего кода по мере его написания, тогда полную картину видеть гораздо проще (особенно это помогает со старым кодом)
            std::string method_name;
            std::string log_prefix;
            if (method == 0) {
                method_name = "ORB";
                log_prefix = "[ORB_OCV] ";
                // ORB - один из видов ключевых дескрипторов, отличается высокой скоростью и относительно неплохим качеством
                cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(); // здесь можно было бы поиграть с его параметрами, например выделять больше чем 500 точек, строить большее число ступеней пирамиды и т.п.
                detector->detect(img0, kps0); // детектируем ключевые точки на исходной картинке
                detector->detect(img1, kps1); // детектируем ключевые точки на преобразованной картинке

                detector->compute(img0, kps0, desc0);
                detector->compute(img1, kps1, desc1);
            } else if (method == 1) {
                method_name = "SIFTOCV";
                log_prefix = "[SIFTOCV] ";
                // ORB - один из видов ключевых дескрипторов, отличается высокой скоростью и относительно неплохим качеством
                cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create(); // здесь можно было бы поиграть с его параметрами, например выделять больше чем 500 точек, строить большее число ступеней пирамиды и т.п.
                detector->detect(img0, kps0); // детектируем ключевые точки на исходной картинке
                detector->detect(img1, kps1); // детектируем ключевые точки на преобразованной картинке

                detector->compute(img0, kps0, desc0);
                detector->compute(img1, kps1, desc1);
            } else if (method == 2) {
                //TODO remove 'return' and uncomment
                //return;
                method_name = "SIFT_MY";
                log_prefix = "[SIFT_MY] ";
                phg::SIFT mySIFT;
                mySIFT.detectAndCompute(img0, kps0, desc0);
                mySIFT.detectAndCompute(img1, kps1, desc1);
            } else {
                rassert(false, 13532513412); // это не проверка как часть тестирования, это проверка что число итераций в цикле и if-else ветки все еще согласованы и не разошлись
            }

            std::cout << log_prefix << "Points detected: " << kps0.size() << " -> " << kps1.size() << " (in " << t.elapsed() << " sec)" << std::endl;
    
            std::vector<cv::Point2f> ps01(kps0.size()); // давайте построим эталон - найдем куда бы должны были сместиться ключевые точки с исходного изображения с учетом нашей матрицы трансформации M
            {
                std::vector<cv::Point2f> ps0(kps0.size()); // здесь мы сейчас расположим детектированные ключевые точки (каждую нужно преобразовать из типа КлючеваяТочка в Точка2Дэ)
                for (size_t i = 0; i < kps0.size(); ++i) {
                    ps0[i] = kps0[i].pt;
                }
                cv::transform(ps0, ps01, M); // преобразовываем все точки с исходного изображения в систему координат его искаженной версии с учетом матрицы M, эти точки - эталон
            }

            double error_sum = 0.0;          // считаем суммарную ошибку координат сопоставлений точек чтобы найти среднюю ошибку (в пикселях)
            double size_ratio_sum = 0.0;     // хотим найти среднее соотношение размера сопоставленных ключевых точек (чтобы сверить эту пропорцию с тестируемым перепадом масштаба)
            double angle_diff_sum = 0.0;     // хотим найти среднее отличие угла наклона сопоставленных ключевых точек (чтобы сверить этот угол с тестируемым в тестах поворотом)
            double desc_dist_sum = 0.0;      // хотим найти среднее расстояние между дескрипторами сопоставленных ключевых точек
            double desc_rand_dist_sum = 0.0; // найдем среднее расстояние между случайными парами ключевых точек (чтобы было с чем сравнить расстояние сопоставленных точек)
            size_t n_matched = 0;   // число успешно сопоставившихся исходных точек
            size_t n_in_bounds = 0; // число исходных точек которые после преобразования координат не вышли за пределы картинки (т.е. в целом имели шансы на успешное сопоставление)
            std::vector<unsigned char> is_not_matched0(kps0.size(), true); // для каждой исходной точки хотим понять сопоставилась ли она
            std::vector<unsigned char> is_not_matched1(kps1.size(), true); // для каждой точки с результирующей картинки хотим понять сопоставился ли с ней хоть кто-то

            // эта прагма - способ распараллелить цикл на все ядра процессора (см. OpenMP parallel for)
            // reduction позволяет сказать OpenMP что нужно провести редукцию суммированием для каждой из переменных: error_sum, n_matched, n_in_bounds, ...
            // мы ведь хотим найти сумму по всем потокам
            #pragma omp parallel for reduction(+:error_sum, n_matched, n_in_bounds, size_ratio_sum, angle_diff_sum, desc_dist_sum, desc_rand_dist_sum)
            for (ptrdiff_t i = 0; i < kps0.size(); ++i) {
                cv::Point2f p01 = ps01[i]; // взяли ожидаемую координату куда должна была перейти точка
                if (p01.x > 0 && p01.x < width && p01.y > 0 && p01.y < height) {
                    n_in_bounds += 1; // засчитали точку как "не вышла за пределы картинки - имеет шансы на успешное сопоставление"
                } else {
                    continue;
                }

                ptrdiff_t closest_j = -1; // будем искать ближайшую точку детектированную на искаженном изображении 
                double min_error = std::numeric_limits<float>::max();
                for (ptrdiff_t j = 0; j < kps1.size(); ++j) {
                    double error = cv::norm(kps1[j].pt - p01);
                    if (error < min_error) {
                        min_error = error;
                        closest_j = j;
                    }
                }
                if (closest_j != -1 && min_error <= MAX_ACCEPTED_PIXEL_ERROR*width) {
                    // мы нашли что-то достаточно близкое - успех!
                    is_not_matched0[i] = false;
                    is_not_matched1[closest_j] = false;
                    ++n_matched;
                    error_sum += min_error;
                    if (kps0[i].size != 0.0) {
                        size_ratio_sum += kps1[closest_j].size / kps0[i].size;
                    }
                    angle_diff_sum += diffAngles(kps0[i].angle, kps1[closest_j].angle);

                    cv::Mat d0 = desc0.rowRange(cv::Range(i, i + 1));
                    cv::Mat d1 = desc1.rowRange(cv::Range(closest_j, closest_j + 1));
                    size_t random_j = (239017 * i + 1232142) % kps1.size();
                    cv::Mat random_d1 = desc1.rowRange(cv::Range(random_j, random_j + 1));;
                    if (method_name == "ORB") {
                        desc_rand_dist_sum += cv::norm(d0, random_d1, cv::NORM_HAMMING);

                        desc_dist_sum += cv::norm(d0, d1, cv::NORM_HAMMING);
                    } else if (method_name == "SIFTOCV" || method_name == "SIFT_MY") {
                        desc_rand_dist_sum += cv::norm(d0, random_d1, cv::NORM_L2);

                        desc_dist_sum += cv::norm(d0, d1, cv::NORM_L2);
                        
                        // Это способ заглянуть в черную коробку, так вы можете визуально посмотреть на то
                        // что за числа в дескрипторах двух сопоставленных точек, насколько они похожи,
                        // и сверить что расстояние между дескрипторами - это действительно расстояние
                        // между точками в пространстве высокой размерности:
#if 0
                        if (i % 100 == 0) {
                            #pragma omp critical
                            {
                                std::cout << "d0: " << d0 << std::endl;
                                std::cout << "d1: " << d1 << std::endl;
                                std::cout << "d1-d0: " << d1-d0 << std::endl;
                                cv::Mat mul;
                                cv::multiply((d1-d0), (d1-d0), mul);
                                std::cout << "(d1-d0)^2: " << mul << std::endl;
                                std::cout << "sum((d1-d0)^2): " << cv::sum(mul) << std::endl;
                                std::cout << "sqrt(sum((d1-d0)^2)): " << sqrt(cv::sum(mul)[0]) << std::endl;
                                std::cout << "norm: " << cv::norm(d0, d1, cv::NORM_L2) << std::endl;
                            }
                        }
#endif
                    }
                }
            }
            rassert(n_matched > 0, 2319241421512); // это не проверка как часть тестирования, это проверка что я не набагал и что дальше не будет деления на ноль :)
            double recall = n_matched*1.0 / n_in_bounds;
            double avg_error = error_sum / n_matched;
            std::cout << log_prefix << n_matched << "/" << n_in_bounds << " (recall=" << recall << ") with average error=" << avg_error << std::endl;
            std::cout << log_prefix << "average size ratio between matched points: " << (size_ratio_sum / n_matched) << std::endl;
            if (angle_diff_sum != 0.0) {
                std::cout << log_prefix << "average angle difference between matched points: " << (angle_diff_sum / n_matched) << " degrees" << std::endl;
                // TODO почему SIFT менее точно угадывает средний угол отклонения? изменяется ли ситуация если выкрутить параметр ORIENTATION_VOTES_PEAK_RATIO=0.999? почему?
            }
            if (desc_dist_sum != 0.0 && desc_rand_dist_sum != 0.0) {
                std::cout << log_prefix << "average descriptor distance between matched points: " << (desc_dist_sum / n_matched) << " (random distance: " << (desc_rand_dist_sum / n_matched) << ") => differentiability=" << (desc_dist_sum / desc_rand_dist_sum) << std::endl;
            }

            // а вот это проверка качества, самая важная часть теста, проверяем насколько часто одни и те же характерные точки детектируются
            // несмотря на несущественное искажение изображения
            // т.е. мы по сути проверяем что "ключевые точки детектируются инвариантно к смещению, повороту и масштабу"
            EXPECT_GT(recall, minRecall);
            // и проверяем среднюю ошибку в пикселях
            EXPECT_LT(avg_error, MAX_AVG_PIXEL_ERROR*width);

            cv::Mat result0 = img0.clone();
            cv::Mat result1 = img1.clone();
            // рисует отладочные картинки, это удобно делать по коду вообще везде, чтобы легко и удобно всегда было заглянуть в черную коробку чтобы попробовать понять
            // где проблемы, или где можно что-то улучшить
            drawKeyPoints(result0, kps0, is_not_matched0);
            drawKeyPoints(result1, kps1, is_not_matched1);
    
            cv::Mat result = concatenateImagesLeftRight(result0, result1);
            cv::putText(result, log_prefix + " recall=" + to_string(recall), cv::Point(10, 30), cv::FONT_HERSHEY_DUPLEX, 0.75, CV_RGB(255, 255, 0));
            cv::putText(result, "avgPixelsError=" + to_string(avg_error), cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 0.75, CV_RGB(255, 255, 0));

            // отладочную визуализацию сохраняем в папку чтобы легко было посмотреть на любой промежуточный результат
            // или в данном случае - на любой результат любого теста
            cv::imwrite("data/debug/test_sift/" + getTestSuiteName() + "/" + getTestName() + "_" + method_name + ".png", result);

            if (SHOW_RESULTS) {
                // показать результат сразу в диалоге удобно если вы запускаете один и тот же тест раз за разом
                // и хотите сразу видеть результат чтобы его оценить, вместо того чтобы идти в папочку и кликать по файлу
                cv::imshow("Red thick circles - not matched", result);
                cv::waitKey();
            }
        }
    }
}

// создаем матрицу описывающую преобразование пространства "сдвиг на вектор"
cv::Mat createTranslationMatrix(double dx, double dy) {
// [1, 0, dx]
// [0, 1, dy]
    cv::Mat M = cv::Mat(2, 3, CV_64FC1, 0.0);
    M.at<double>(0, 0) = 1.0;
    M.at<double>(1, 1) = 1.0;
    M.at<double>(0, 2) = dx;
    M.at<double>(1, 2) = dy;
    return M;
}


TEST (SIFT, MovedTheSameImage) {
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(0.0, 0.0), minRecall);
}

TEST (SIFT, MovedImageRight) {
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(50.0, 0.0), minRecall);
}

TEST (SIFT, MovedImageLeft) {
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(-50.0, 0.0), minRecall);
}

TEST (SIFT, MovedImageUpHalfPixel) {
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(0.0, -50.5), minRecall);
}

TEST (SIFT, MovedImageDownHalfPixel) {
    double minRecall = 0.75;
    evaluateDetection(createTranslationMatrix(0.0, 50.5), minRecall);
}

TEST (SIFT, Rotate10) {
    double angleDegreesClockwise = 10;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Rotate20) {
    double angleDegreesClockwise = 20;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Rotate30) {
    double angleDegreesClockwise = 30;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Rotate40) {
    double angleDegreesClockwise = 40;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Rotate45) {
    double angleDegreesClockwise = 45;
    double scale = 1.0;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Rotate90) {
    double angleDegreesClockwise = 90;
    double scale = 1.0;
    double minRecall = 0.75;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Scale50) {
    double angleDegreesClockwise = 0;
    double scale = 0.5;
    double minRecall = 0.40;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Scale70) {
    double angleDegreesClockwise = 0;
    double scale = 0.7;
    double minRecall = 0.40;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Scale90) {
    double angleDegreesClockwise = 0;
    double scale = 0.9;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Scale110) {
    double angleDegreesClockwise = 0;
    double scale = 1.1;
    double minRecall = 0.60;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Scale130) {
    double angleDegreesClockwise = 0;
    double scale = 1.3;
    double minRecall = 0.50;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Scale150) {
    double angleDegreesClockwise = 0;
    double scale = 1.5;
    double minRecall = 0.50;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Scale175) {
    double angleDegreesClockwise = 0;
    double scale = 1.75;
    double minRecall = 0.75;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), 0.3);
}

TEST (SIFT, Scale200) {
    double angleDegreesClockwise = 0;
    double scale = 2.0;
    double minRecall = 0.20;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Rotate10Scale90) {
    double angleDegreesClockwise = 10;
    double scale = 0.9;
    double minRecall = 0.65;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, Rotate30Scale75) {
    double angleDegreesClockwise = 30;
    double scale = 0.75;
    double minRecall = 0.50;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(200, 256), -angleDegreesClockwise, scale), minRecall);
}

TEST (SIFT, HerzJesu19RotateM40) {
    cv::Mat jesu19 = cv::imread("data/src/test_sift/herzjesu19.png");

    ASSERT_FALSE(jesu19.empty()); // проверка что картинка была загружена
    // убедитесь что рабочая папка (Edit Configurations...->Working directory) указывает на корневую папку проекта

    double angleDegreesClockwise = -40;
    double scale = 1.0;
    double minRecall = 0.75;
    evaluateDetection(cv::getRotationMatrix2D(cv::Point(jesu19.cols/2, jesu19.rows/2), -angleDegreesClockwise, scale), minRecall, jesu19);
}
