#include "bruteforce_matcher_gpu.h"

#include <iostream>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/rasserts.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt
#include "cl/bruteforce_matcher_cl.h"

#define BF_MATCHER_GPU_VERBOSE 0

void phg::BruteforceMatcherGPU::train(const cv::Mat &train_desc)
{
    if (train_desc.rows < 2) {
        throw std::runtime_error("BruteforceMatcher:: train : needed at least 2 train descriptors");
    }

    train_desc_ptr = &train_desc;
}

void phg::BruteforceMatcherGPU::knnMatch(const cv::Mat &query_desc,
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

    gpu::Device device = gpu::chooseDevice(BF_MATCHER_GPU_VERBOSE);
    if (!device.supports_opencl) {
        throw std::runtime_error("No OpenCL device found");
    }

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    rassert(train_desc_ptr->type() == CV_32FC1, 23412414126777);
    rassert(query_desc.type() == CV_32FC1, 23412414126777);

    const int ndim = query_desc.cols;
    rassert(ndim == train_desc_ptr->cols, 353635235225);

    const int ndesc = query_desc.rows;
    const int n_train_desc = train_desc_ptr->rows;

    timer t;
    gpu::gpu_mem_32f train_data, query_data;
    gpu::gpu_mem_32f res_matches_distance;
    gpu::gpu_mem_32u res_matches_train_idx, res_matches_query_idx;

    train_data.resizeN(n_train_desc * ndim);  // массивы в видеопамяти с дескрипторами (выложенными подряд)
    query_data.resizeN(ndesc * ndim);         // массивы в видеопамяти с дескрипторами (выложенными подряд)
    res_matches_distance.resizeN(ndesc * 2);  // найденные расстояния лучших 2 сопоставлений
    res_matches_train_idx.resizeN(ndesc * 2); // найденные индексы двух лучших сопоставленных пар (в списке train ключевых точек)
    res_matches_query_idx.resizeN(ndesc * 2); // найденные индексы двух лучших сопоставленных пар (в списке query ключевых точек)
    rassert(train_desc_ptr->isContinuous(), 352365262346252);
    rassert(query_desc.isContinuous(), 352365262346252);
    train_data.write(train_desc_ptr->ptr(), train_data.size()); // прогрузили дескрипторы в видеопамять
    query_data.write(query_desc.ptr(), query_data.size());      // прогрузили дескрипторы в видеопамять

    if (BF_MATCHER_GPU_VERBOSE) std::cout << "[BFMatcher] data allocated and loaded in " << t.elapsed() << " s" << std::endl;

    t.restart();
    const unsigned int keypoints_per_wg = 4;
    std::string kernel_defines = "-D KEYPOINTS_PER_WG=" + to_string(keypoints_per_wg);
    ocl::Kernel bruteforce_matcher(bruteforce_matcher_kernel, bruteforce_matcher_kernel_length, "bruteforce_matcher", kernel_defines);
    bruteforce_matcher.compile(BF_MATCHER_GPU_VERBOSE);
    if (BF_MATCHER_GPU_VERBOSE) std::cout << "[BFMatcher] kernel compiled in " << t.elapsed() << " s" << std::endl;

    t.restart();
    unsigned int work_group_size = 128;
    rassert(work_group_size == ndim, 3541414124125125); // мы полагаемся на то что один поток рабочей группы будет грузить одно значение из дескриптора
    unsigned int global_work_size = (ndesc + keypoints_per_wg - 1) / keypoints_per_wg; // каждая рабочая группа обрабатывает keypoints_per_wg=4 дескриптора из query (сопоставляет их со всеми train)
    gpu::WorkSize ws(work_group_size, 1,
                     work_group_size, global_work_size);
    bruteforce_matcher.exec(ws,
                            train_data, query_data,
                            res_matches_train_idx, res_matches_query_idx, res_matches_distance,
                            n_train_desc, ndesc);
    if (BF_MATCHER_GPU_VERBOSE) std::cout << "[BFMatcher] kernel executed in " << t.elapsed() << " s" << std::endl;

    t.restart();
    std::vector<float> distance_res(ndesc * 2, std::numeric_limits<float>::max());
    std::vector<unsigned int> train_idx_res(ndesc * 2, std::numeric_limits<unsigned int>::max());
    std::vector<unsigned int> query_idx_res(ndesc * 2, std::numeric_limits<unsigned int>::max());
    res_matches_distance.readN(distance_res.data(), ndesc * 2);
    res_matches_train_idx.readN(train_idx_res.data(), ndesc * 2);
    res_matches_query_idx.readN(query_idx_res.data(), ndesc * 2);
    if (BF_MATCHER_GPU_VERBOSE) std::cout << "[BFMatcher] result data loaded in " << t.elapsed() << " s" << std::endl;

    t.restart();
    matches.resize(ndesc);
    for (int qi = 0; qi < ndesc; ++qi) {
        std::vector<cv::DMatch> &dst = matches[qi];
        dst.resize(2);

        for (int ki = 0; ki < 2; ++ki) {
            cv::DMatch match;
            match.distance = distance_res[qi * 2 + ki];
            match.imgIdx = 0;
            match.queryIdx = query_idx_res[qi * 2 + ki];
            match.trainIdx = train_idx_res[qi * 2 + ki];
            if (!(match.queryIdx == qi)) {
                std::cerr << match.queryIdx << " != " << qi << std::endl;
            }
            rassert(match.queryIdx == qi, 345151241241251);
            dst[ki] = match;
        }
        rassert(dst[0].distance <= dst[1].distance, 645151255341241251);
    }
    if (BF_MATCHER_GPU_VERBOSE) std::cout << "[BFMatcher] data unpacked in " << t.elapsed() << " s" << std::endl;
}
