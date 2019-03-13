#include "mat_util.hpp"

namespace AOG_LIB {
namespace SAOT {
namespace UTIL {

void MatlabColonExpression(const int start, const int end, const int step, std::vector<int>& output) {
    int current = start;
    while (current < end) {
        output.push_back(current);
        current += step;
    }
}

void MatlabColonExpression(const float start, const float end, const float step, std::vector<float>& output) {
    float current = start;
    while (current < end) {
        output.push_back(current);
        current += step;
    }
}

void MatlabFind(const std::vector<int> v, std::vector<int> &output) {
    output.clear();
    for (int i = 0; i < v.size(); i++) {
        if (v[i] != 0) output.push_back(i);
    }
}

cv::Mat MatlabColonExpression(const int start, const int end, const int step) {
    int size = (int)((end - start + 1) / step);
    cv::Mat mat = cv::Mat(1, size, CV_32S, cv::Scalar(0));
    for (int i = 0; i < size; i++) mat.at<int>(0, i) = start + i * step;
    return mat;
}

}  // namespace UTIL
}  // namespace SAOT
}  // namespace AOG_LIB
