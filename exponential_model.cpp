#include "exponential_model.hpp"
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <cmath>
#include <ctime>
#include "filter.hpp"
#include "misc.hpp"
#include "util/file_util.hpp"
#include "util/meta_type.hpp"

namespace AOG_LIB {
namespace SAOT {

void ExponentialModel::Build() {
  std::vector<std::string> img_list;
  UTIL::GetFileList(img_list, img_dir_, img_ext_, /*fullpath=*/true);
  int num_image = img_list.size();
  std::vector<cv::Size> img_size;
  img_size.reserve(num_image);
  MatCell_1<cv::Mat> images = CreateMatCell1Dim<cv::Mat>(num_image);
  int sizex, sizey;
  for (int i = 0; i < num_image; i++) {
    cv::Mat img = cv::imread(img_list[i], cv::IMREAD_GRAYSCALE);
    images[i] = img;
    img_size[i] = img.size();
    if (i == 0) {
      sizex = img.rows;
      sizey = img.cols;
    } else {
      sizex = fmin(sizex, img.rows);
      sizey = fmin(sizey, img.cols);
    }
  }
  for (int i = 0; i < num_image; i++) {
    images[i] = images[i].rowRange(0, sizex).colRange(0, sizey);
  }
  // filtering background images
  BOOST_LOG_TRIVIAL(debug) << "Start filtering";
  std::clock_t start_time = std::clock();
  MakeFilter(scale_filter_, num_orient_, all_filter, all_symbol);
  // half size of gabor
  int half_filter_size = all_filter[0].rows / 2;
  // ApplyFilterfftSame
  MatCell_2<cv::Mat> filtered_images;
  ApplyFilterfft(config_, images, all_filter, filtered_images);
  double filter_time = (std::clock() - start_time) / CLOCKS_PER_SEC;
  BOOST_LOG_TRIVIAL(debug) << "filtering time: " << filter_time << " seconds";
  // compute hisogram of q()
  BOOST_LOG_TRIVIAL(debug) << "Start histogramming";
  start_time = std::clock();
  int num_bins = static_cast<int>(floor(saturation_ / bin_size_) + 1);
  cv::Mat histog = cv::Mat::zeros(num_bins, 1, CV_64F);

  Histogram(filtered_images, config_, cv::Size(sizex, sizey), bin_size_,
            num_bins, histog);
  double hist_time = (std::clock() - start_time) / CLOCKS_PER_SEC;
  BOOST_LOG_TRIVIAL(debug) << "histogramming time: " << hist_time << " seconds";
  // compute stored lambda, expectation, logZ
  stored_param = CreateMatCell1Dim<ExpParam>(num_stored_point_);
  cv::Mat r(num_bins, 1, CV_64F);
  for (int i = 0; i < num_bins; i++) {
    r.at<double>(i, 0) = i * bin_size_;
  }
  double lambda, Z;
  for (int k = 0; k < num_stored_point_; k++) {
    lambda = (double)k / 10.0;

    cv::Mat p(num_bins, 1, CV_64F);
    for (int i = 0; i < num_bins; i++) {
      p.at<double>(i, 0) =
          exp(lambda * r.at<double>(i, 0)) * histog.at<double>(i, 0);
    }
    Z = cv::sum(p)[0];
    p = p / Z;
    stored_param[k].storedExpectation =
        cv::sum(cv::Mat(r.mul(p) * bin_size_))[0];
    stored_param[k].storedlambda = lambda;
    stored_param[k].storedlogZ = log(Z);
  }

  CorrFilter(all_filter, epsilon_, correlation);
}

}  // namespace SAOT
}  // namespace AOG_LIB