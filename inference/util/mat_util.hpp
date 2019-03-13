#ifndef SAOT_UTIL_MAT_UTIL_HPP_
#define SAOT_UTIL_MAT_UTIL_HPP_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

/**
 * \author Tengyu Liu
*/
namespace AOG_LIB {
namespace SAOT {
namespace UTIL {

void MatlabColonExpression(const int start, const int end, const int step, std::vector<int>& output);

void MatlabColonExpression(const float start, const float end, const float step, std::vector<float>& output);

cv::Mat MatlabColonExpression(const int start, const int end, const int step);

void MatlabFind(const std::vector<int> v, std::vector<int> &output);

}  // namespace UTIL
}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_UTIL_VIZ_UTIL_HPP_