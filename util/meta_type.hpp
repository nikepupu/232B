#ifndef SAOT_UTIL_META_TYPE_HPP_
#define SAOT_UTIL_META_TYPE_HPP_

#include <boost/multi_array.hpp>
#include <opencv2/core/core.hpp>
#include <cassert>

#define PI 3.1415926

namespace AOG_LIB {
namespace SAOT {

struct template_filter
{
  float row, col, ori, scale, ind, trans, lambda, logZ;
};

template<typename T>
using MatCell_1 = boost::multi_array<T, 1>;

template<typename T>
using MatCell_2 = boost::multi_array<T, 2>;

template<typename T>
using MatCell_3 = boost::multi_array<T, 3>;


template<typename T>
inline MatCell_1<T> CreateMatCell1Dim(int length) {
  MatCell_1<T> new_cell(boost::extents[length]);
  return new_cell;
}

template<typename T>
inline MatCell_2<T> CreateMatCell2Dim(int row, int col) {
  MatCell_2<T> new_cell(boost::extents[row][col]);
  return new_cell;
}

template<typename T>
inline MatCell_3<T> CreateMatCell3Dim(int d1, int d2, int d3) {
  MatCell_3<T> new_cell(boost::extents[d1][d2][d3]);
  return new_cell;
}

}
}

#endif  // SAOT_UTIL_META_TYPE_HPP_
