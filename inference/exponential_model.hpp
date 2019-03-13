#ifndef SAOT_EXPONENTIAL_MODEL_HPP_
#define SAOT_EXPONENTIAL_MODEL_HPP_

#ifndef BOOST_LOG_DYN_LINK
#define BOOST_LOG_DYN_LINK 1
#endif


#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "util/meta_type.hpp"
#include "saot_config.hpp"
/**
 * \author Zilong Zheng
 */
namespace AOG_LIB {
namespace SAOT {

struct ExpParam {
  double storedlambda, storedExpectation, storedlogZ;
};

class ExponentialModel {
 public:
  ExponentialModel(const SAOTConfig& config, const std::string& img_dir,
                   const std::string& img_ext)
      : config_(config),
        img_dir_(img_dir),
        img_ext_(img_ext),
        scale_filter_(config.scale_filter),
        num_orient_(config.num_orient),
        epsilon_(config.epsilon),
        saturation_(config.saturation),
        bin_size_(config.bin_size),
        num_stored_point_(config.num_stored_point) {}

  void SetBackgroudnImageDir(const std::string& img_dir);

  void Build();

  MatCell_1<cv::Mat> all_filter, all_symbol;

  MatCell_1<ExpParam> stored_param;

  MatCell_2<cv::Mat> correlation;

 private:
  SAOTConfig config_;
  double scale_filter_;
  int num_orient_, num_stored_point_;
  double epsilon_, saturation_, bin_size_;
  //   Background image dir
  std::string img_dir_, img_ext_;
};

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_EXPONENTIAL_MODEL_HPP_
