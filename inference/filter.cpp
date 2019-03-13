#include "util/meta_type.hpp"
#include "filter.hpp"
#include <cmath>
#include "misc.hpp"
#include "saot_config.hpp"


namespace AOG_LIB {
namespace SAOT {

void GaborFilter(int scale, int orient, cv::Mat &filter, cv::Mat &symbol) {
  int expand = 12;
  int h = static_cast<int>(floor(scale * expand + 0.5));
  double alpha = M_PI * orient / 180;
  cv::Mat Gauss = cv::Mat::zeros(h + h + 1, h + h + 1, CV_64F);
  cv::Mat Gcos, Gsin;
  Gcos = cv::Mat::zeros(h + h + 1, h + h + 1, CV_64F);
  Gsin = cv::Mat::zeros(h + h + 1, h + h + 1, CV_64F);
  symbol = cv::Mat::zeros(h + h + 1, h + h + 1, CV_64F);
  double s = 0, sc = 0;
  double Scos = 0, Ssin = 0;
  for (int x0 = -h; x0 <= h; x0++) {
    for (int y0 = -h; y0 <= h; y0++) {
      int inCircle;
      if (x0 * x0 + y0 * y0 > h * h) {
        inCircle = 0;
      } else {
        inCircle = 1;
      }
      double x = (x0 * cos(alpha) + y0 * sin(alpha)) / scale;
      double y = (y0 * cos(alpha) - x0 * sin(alpha)) / scale;
      double g = exp(-(4 * x * x + y * y) / 100) / 50 / M_PI / (scale * scale);
      Gauss.at<double>(h + x0, h + y0) = g * inCircle;
      Gcos.at<double>(h + x0, h + y0) = g * cos(x) * inCircle;
      Gsin.at<double>(h + x0, h + y0) = g * sin(x) * inCircle;
      s += Gauss.at<double>(h + x0, h + y0);
      sc += Gcos.at<double>(h + x0, h + y0);
      Scos += Gcos.at<double>(h + x0, h + y0) * Gcos.at<double>(h + x0, h + y0);
      Ssin += Gsin.at<double>(h + x0, h + y0) * Gsin.at<double>(h + x0, h + y0);
      if (fabs(x) < 3.4 && inCircle) {
        symbol.at<double>(h + x0, h + y0) = 1;
      } else {
        symbol.at<double>(h + x0, h + y0) = 0;
      }
    }
  }
  double r = sc / s;
  Scos = sqrt(Scos);
  Ssin = sqrt(Ssin);
  for (int x0 = -h; x0 <= h; x0++) {
    for (int y0 = -h; y0 <= h; y0++) {
      Gcos.at<double>(h + x0, h + y0) = Gcos.at<double>(h + x0, h + y0) -
                                        Gauss.at<double>(h + x0, h + y0) * r;
      Gcos.at<double>(h + x0, h + y0) = Gcos.at<double>(h + x0, h + y0) / Scos;
      Gsin.at<double>(h + x0, h + y0) = Gsin.at<double>(h + x0, h + y0) / Ssin;
    }
  }
  cv::Mat G[2];
  G[0] = Gcos;
  G[1] = Gsin;
  cv::merge(G, 2, filter);
}

void MakeFilter(int scale, int num_orient, MatCell_1<cv::Mat> &all_filter,
                MatCell_1<cv::Mat> &all_symbol) {
  all_filter = CreateMatCell1Dim<cv::Mat>(num_orient);
  all_symbol = CreateMatCell1Dim<cv::Mat>(num_orient);
  for (int o = 0; o < num_orient; o++) {
    int orient = o * 180 / num_orient;
    cv::Mat filter, symbol;
    GaborFilter(scale, orient, filter, symbol);
    all_filter[o] = filter;
    all_symbol[o] = symbol;
  }
}

void CorrFilter(const MatCell_1<cv::Mat> &filters, double epsilon,
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

void ApplyFilterfft(const SAOTConfig config, const MatCell_1<cv::Mat> &images,
                    const MatCell_1<cv::Mat> &filters,
                    MatCell_2<cv::Mat> &filtered_images) {
  int num_images = images.shape()[0];
  int num_filters = filters.shape()[0];
  int h = (filters[0].rows - 1) / 2;

  for (int i = 0; i < num_images; i++) {
    double tot = 0.0;
    int sx = images[i].rows, sy = images[i].cols;
    cv::Mat fftI;
    cv::copyMakeBorder(images[i], fftI, h, h, h, h, cv::BORDER_CONSTANT, 0);
    cv::dft(fftI, fftI);
    MatCell_1<cv::Mat> filtered_results =
        CreateMatCell1Dim<cv::Mat>(num_filters);
    for (int j = 0; j < num_filters; j++) {
      cv::Mat out;
      cv::copyMakeBorder(filters[j], out, h, h, h, h, cv::BORDER_CONSTANT, 0);
      out.convertTo(out, CV_32F);
      cv::dft(out, out, cv::DFT_INVERSE);
      cv::Mat filtered = out.rowRange(h, h + sx).colRange(h, h + sy);
      //Compute local energy
      cv::Mat energy = cv::abs(filtered);
      energy.rowRange(0, h).setTo(cv::Scalar(0.0));
      energy.rowRange(sx - h, sx).setTo(cv::Scalar(0.0));
      energy.colRange(0, h).setTo(cv::Scalar(0.0));
      energy.colRange(sy - h, sy).setTo(cv::Scalar(0.0));
      filtered_results[j] = energy;
      tot +=
          cv::sum(energy.rowRange(h, sx - h - 1).colRange(h, sy - h - 1))[0] /
          (sx - 2 * h - 1) / (sy - 2 * h - 1);
    }
    double ave = tot / num_filters;
    for (int j = 0; j < num_filters; j++) {
      filtered_results[j] /= ave;
    }
    LocalNormalize(cv::Size(sx, sy), num_filters, h,
                   cv::Size(config.local_half_x, config.local_half_y),
                   config.threshold_factor, filtered_results);
    filtered_images[i] = filtered_results;
  }
}

}  // namespace SAOT
}  // namespace AOG_LIB
