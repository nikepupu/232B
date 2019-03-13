#ifndef SAOT_FILTER_HPP_
#define SAOT_FILTER_HPP_

#include "saot_config.hpp"
#include "util/meta_type.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <vector>

namespace AOG_LIB {
namespace SAOT {
/**
 * \author Zilong Zheng
*/
void GaborFilter(int scale, int orient, cv::Mat &filter, cv::Mat &symbol);

void MakeFilter(int scale, int num_orient, MatCell_1<cv::Mat> &all_filter,
                MatCell_1<cv::Mat> &all_symbol);

// Compute inhibition map among filters
void CorrFilter(const MatCell_1<cv::Mat> &filters, double epsilon,
                MatCell_2<cv::Mat> &inhibit_map);

void ApplyFilterfft(const SAOTConfig config, const MatCell_1<cv::Mat> &images,
                    const MatCell_1<cv::Mat> &filters,
                    MatCell_2<cv::Mat> &filtered_images);

void ApplyFilterfftSame(const MatCell_1<cv::Mat> &images,
                        const MatCell_1<cv::Mat> &filters,
                        MatCell_2<cv::Mat> &filtered_images);

} // namespace SAOT
} // namespace AOG_LIB

#endif // SAOT_FILTER_HPP_