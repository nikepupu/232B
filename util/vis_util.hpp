#ifndef SAOT_UTIL_VIZ_UTIL_HPP_
#define SAOT_UTIL_VIZ_UTIL_HPP_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

/**
 * \author Xiaofeng Gao
*/
namespace AOG_LIB {
namespace SAOT {
namespace UTIL {

void DisplayImages(const std::vector<cv::Mat>& images);
void DisplayMatchedTemplate(const std::vector<cv::Mat>& images);
void DisplayTemplate(const std::vector<cv::Mat>& images);
void DrawGaborSymbol(const std::vector<cv::Mat>& images);

}  // namespace UTIL
}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_UTIL_VIZ_UTIL_HPP_