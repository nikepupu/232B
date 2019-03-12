#ifndef SAOT_SAOT_INFERENCE_CONFIG_HPP_
#define SAOT_SAOT_INFERENCE_CONFIG_HPP_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "util/meta_type.hpp"

namespace AOG_LIB {
namespace SAOT{

struct SAOTInferenceConfig {
//from the first file
  std::string category;
  std::string img_dir;
  std::string img_ext;
  std::string output_dir;
  std::string bkg_img_dir;
  double resize_factor;
  int num_resolution;
  int template_size[2], part_size[2];
  std::vector<int> part_rotation_range;
  int num_part_rotation;
  int max_part_relative_rotation;
  double min_rotation_dif;
  std::vector<int> rotation_range;
  int num_rotate;
  int num_element;
  double location_perturb_fraction;
  int location_shift_limit;
  int subsample_S2, subsample_M2, subsample_S3 = 1;
  int part_margin[2];
  double epsilon;
  int subsample;
  int num_image;
  int num_orient;
  int num_scale;
  int orient_shift_limit;
  std::vector<std::string> image_name;
  int local_half_x, local_half_y;
  int half_filter_size;
  double threshold_factor;
  double saturation;
  int numStoredPoint; /* number of stored points of lambda in exponential model */   
  int num_iteration;
  int numCandPart;


  int partSizeX;
  int partSizeY;
  std::vector<int> PartLocX,PartLocY;


// the second file
MatCell_2<cv::Mat> allSelectedx;
MatCell_2<cv::Mat> allSelectedy;
MatCell_2<cv::Mat> allSelectedOrient;
MatCell_1<cv::Mat> selectedlambda;
MatCell_1<cv::Mat> selectedLogZ;
MatCell_1<cv::Mat> allSymbol;

MatCell_2<cv::Mat> largerAllSelectedx;
MatCell_2<cv::Mat> largerAllSelectedy;
MatCell_2<cv::Mat> largerAllSelectedOrient;
MatCell_1<cv::Mat> largerSelectedlambda;
MatCell_1<cv::Mat> largerSelectedLogZ;

//thrid file
int *PartOnOff;
cv::Mat allS3SelectedRow;
cv::Mat allS3SelectedCol;
cv::Mat allS3SelectedOri;

cv::Mat selectedPart;



};

bool LoadSAOTInferenceConfigFile(const std::string &filename1,  const std::string &filename2, const std::string &filename3, SAOTInferenceConfig &config);

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_SAOT_CONFIG_HPP_
