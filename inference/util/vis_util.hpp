#ifndef SAOT_UTIL_VIZ_UTIL_HPP_
#define SAOT_UTIL_VIZ_UTIL_HPP_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "meta_type.hpp"

/**
 * \author Xiaofeng Gao
*/
namespace AOG_LIB {
namespace SAOT {
namespace UTIL {

void DisplayImages(const MatCell_1<cv::Mat>& images, cv::Mat & out_image, int ncol, int bx, int by, bool normalize);

void DisplayTemplate(const std::vector<cv::Mat>& images);

void DrawGaborSymbol(cv::Mat &im, const MatCell_1<cv::Mat> &allSymbol, double row, double col, double orientationIndex, int nGaborOri, double scaleIndex, double intensity);

void DisplayMatchedTemplate(const cv::Size latticeSize, const std::vector<double> &selectedRow, 
			const std::vector<double> &selectedCol, const std::vector<double> &selectedO, const std::vector<double> &selectedS, const std::vector<double> selectedMean, 
			const MatCell_1<cv::Mat> &allSymbol, const int nGaborOri, cv::Mat &sym);


}  // namespace UTIL
}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_UTIL_VIZ_UTIL_HPP_