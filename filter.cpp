#include "filter.hpp"
#include "misc.hpp"
#include "util/meta_type.hpp"
#include "saot_inference_Config.hpp"

namespace AOG_LIB {
namespace SAOT {

cv::Mat GaborFilter(int scale, int orient) {
  int expand = 12;
  int half_size = scale * expand + 0.5;
  int gabor_size = half_size * 2 + 1;
  cv::Mat gabor[2];
  gabor[0] = cv::getGaborKernel(cv::Size(gabor_size, gabor_size), /*sigma=*/5,
                                /*theta=*/CV_PI * orient / 180, /*lambd=*/scale,
                                /*gamma=*/0.5, /*psi=*/0, CV_64F);
  gabor[1] = cv::getGaborKernel(cv::Size(gabor_size, gabor_size), /*sigma=*/5,
                                /*theta=*/CV_PI * orient / 180, /*lambd=*/scale,
                                /*gamma=*/0.5, /*psi=*/CV_PI / 2, CV_64F);
  cv::Mat gabor_comp;
  cv::merge(gabor, 2, gabor_comp);
  return gabor_comp;
}

void MakeFilter(int scale, int num_orient, MatCell_1<cv::Mat> &filters) {
  filters = CreateMatCell1Dim<cv::Mat>(num_orient);
  for (int o = 0; o < num_orient; o++) {
    filters[o] = GaborFilter(scale, o);
  }
}

void CorrFilter(const MatCell_1<cv::Mat> &filters,
                                double epsilon,
                                MatCell_2<cv::Mat> &inhibit_map) {
  int num_filter = filters.size();
  inhibit_map = CreateMatCell2Dim<cv::Mat>(num_filter, num_filter);
  for (int i = 0; i < num_filter; ++i) {
    int hi = filters[i].rows / 2;
    for (int j = 0; j < num_filter; ++j) {
      int hj = filters[j].rows / 2;
      cv::Mat I;
      cv::copyMakeBorder(filters[i], I, hj, hj, hj, hj, cv::BORDER_CONSTANT, 0);
      cv::Mat I_comp[2], filter_comp[2];
      //  I_comp[0] = Re(I), I_comp[1] = Im(I)
      cv::split(I, I_comp);
      cv::split(filters[j], filter_comp);
      // Convolve F{i} by F{j}
      cv::Mat rr, ri, ir, ii;
      cv::filter2D(I_comp[0], rr, /*ddepth=*/-1, filter_comp[0]);
      cv::filter2D(I_comp[1], rr, /*ddepth=*/-1, filter_comp[0]);
      cv::filter2D(I_comp[0], rr, /*ddepth=*/-1, filter_comp[1]);
      cv::filter2D(I_comp[1], rr, /*ddepth=*/-1, filter_comp[1]);
      cv::Mat corr = rr.mul(rr) + ri.mul(ri) + ir.mul(ir) + ii.mul(ii);
      cv::threshold(corr, corr, epsilon, 1.0, cv::THRESH_BINARY_INV);
      inhibit_map[i][j] = corr;
    }
  }
}

void ApplyFilterfft(const SAOTInferenceConfig config, const MatCell_1<cv::Mat> &images,
                    const MatCell_1<cv::Mat> &filters,
                    MatCell_2<cv::Mat> &filtered_images)
{
  // int num_images = images.shape()[0];
  // int num_filters = filters.shape()[0];
  // int h = (filters[0].rows - 1) / 2;

  // for (int i = 0; i < num_images; i++) {
  //   double tot = 0.0;
  //   int sx = images[i].rows, sy = images[i].cols;
  //   cv::Mat fftI;
  //   cv::copyMakeBorder(images[i], fftI, h, h, h, h, cv::BORDER_CONSTANT, 0);
  //   cv::dft(fftI, fftI);
  //   MatCell_1<cv::Mat> filtered_results = CreateMatCell1Dim<cv::Mat>(num_filters);
  //   for (int j = 0; j < num_filters; j++) {
  //     cv::Mat out;
  //     cv::copyMakeBorder(filters[j], out, h, h, h, h, cv::BORDER_CONSTANT, 0);
  //     cv::dft(out, out, cv::DFT_INVERSE);
  //     cv::Mat filtered = out.rowRange(h, h + sx).colRange(h, h + sy);
  //     //  Compute local energy
  //     cv::Mat energy = cv::abs(filtered);
  //     energy.rowRange(0, h).setTo(cv::Scalar(0.0));
  //     energy.rowRange(sx - h, sx).setTo(cv::Scalar(0.0));
  //     energy.colRange(0, h).setTo(cv::Scalar(0.0));
  //     energy.colRange(sy - h, sy).setTo(cv::Scalar(0.0));
  //     filtered_results[j] = energy;
  //     tot +=
  //         cv::sum(energy.rowRange(h, sx - h - 1).colRange(h, sy - h - 1))[0] /
  //         (sx - 2 * h - 1) / (sy - 2 * h - 1);
  //   }
  //   double ave = tot / num_filters;
  //   for (int j = 0; j < num_filters; j++) {
  //    filtered_results[j] /= ave;
  //   }
  //   LocalNormalize(cv::Size(sx, sy), num_filters, h, cv::Size(config.local_half_x, config.local_half_y),
  //                  config.threshold_factor, filtered_results);
  //   filtered_images[i] = filtered_results;
  // }
}

} // namespace SAOT
} // namespace AOG_LIB
