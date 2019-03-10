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

}  // namespace UTIL
}  // namespace SAOT
}  // namespace AOG_LIB
