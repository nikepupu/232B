#ifndef SAOT_UTIL_META_TYPE_HPP_
#define SAOT_UTIL_META_TYPE_HPP_

#include <boost/multi_array.hpp>
#include <cassert>
#include <opencv2/core/core.hpp>

#define PI 3.1415926

namespace AOG_LIB {
namespace SAOT {

struct template_filter {
  float row, col, ori, scale, ind, trans, lambda, logZ;
};

template <typename T> using MatCell_1 = boost::multi_array<T, 1>;

template <typename T> using MatCell_2 = boost::multi_array<T, 2>;

template <typename T> using MatCell_3 = boost::multi_array<T, 3>;

template <typename T> inline MatCell_1<T> *CreateMatCell1DimPtr(int length) {
  MatCell_1<T> *new_cell = new MatCell_1<T>(boost::extents[length]);
  return new_cell;
}

template <typename T>
inline MatCell_2<T> *CreateMatCell2DimPtr(int row, int col) {
  MatCell_2<T> *new_cell = new MatCell_2<T>(boost::extents[row][col]);
  return new_cell;
}

template <typename T>
inline MatCell_3<T> *CreateMatCell3DimPtr(int d1, int d2, int d3) {
  MatCell_3<T> *new_cell = new MatCell_3<T>(boost::extents[d1][d2][d3]);
  return new_cell;
}

template <typename T> inline MatCell_1<T> CreateMatCell1Dim(int length) {
  MatCell_1<T> new_cell(boost::extents[length]);
  return new_cell;
}

template <typename T> inline MatCell_2<T> CreateMatCell2Dim(int row, int col) {
  MatCell_2<T> new_cell(boost::extents[row][col]);
  return new_cell;
}

template <typename T>
inline MatCell_3<T> CreateMatCell3Dim(int d1, int d2, int d3) {
  MatCell_3<T> new_cell(boost::extents[d1][d2][d3]);
  return new_cell;
}

template <typename T>
inline void CreateMatCell1Dim(MatCell_1<T> &mat_cell, int length) {
  mat_cell.resize(boost::extents[length]);
}

template <typename T>
inline void CreateMatCell2Dim(MatCell_2<T> &mat_cell, int row, int col) {
  mat_cell.resize(boost::extents[row][col]);
}

template <typename T>
inline void CreateMatCell3Dim(MatCell_3<T> &mat_cell, int d1, int d2, int d3) {
  mat_cell.resize(boost::extents[d1][d2][d3]);
}
}
}

#endif // SAOT_UTIL_META_TYPE_HPP_
