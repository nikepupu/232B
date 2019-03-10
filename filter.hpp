#ifndef SAOT_FILTER_HPP_
#define SAOT_FILTER_HPP_

#include "util/meta_type.hpp"
#include "saot_inference_Config.hpp"

#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"

namespace AOG_LIB {
namespace SAOT {
/**
 * \author Zilong Zheng
*/
cv::Mat GaborFilter(int scale, int orient);

void MakeFilter(int scale, int num_orient, MatCell_1<cv::Mat> &filters);

// Compute inhibition map among filters
void CorrFilter(const MatCell_1<cv::Mat> &filters, double epsilon,
                MatCell_2<cv::Mat>& inhibit_map);

void ApplyFilterfft(const SAOTInferenceConfig config, const MatCell_1<cv::Mat> &images,
                    const MatCell_1<cv::Mat> &filters,
                    MatCell_2<cv::Mat> &filtered_images);

void ApplyFilterfftSame(const MatCell_1<cv::Mat> &images,
                        const MatCell_1<cv::Mat> &filters,
                        MatCell_2<cv::Mat> &filtered_images);

} // namespace SAOT
} // namespace AOG_LIB

#endif // SAOT_FILTER_HPP_
