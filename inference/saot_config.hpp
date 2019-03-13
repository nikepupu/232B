#ifndef SAOT_SAOT_CONFIG_HPP_
#define SAOT_SAOT_CONFIG_HPP_

#define PartLocX part_loc_x
#define PartLocY part_loc_y 
#define partSizeX part_size[0]
#define partSizeY part_size[1]

#include <string>
#include <vector>

namespace AOG_LIB {
namespace SAOT {

struct SAOTConfig {
  std::string category;
  std::string img_dir;
  std::string img_ext;
  std::string output_dir;
  std::string bkg_img_dir;
  double resize_factor;
  int num_resolution;
  int original_resolution;
  int template_size[2], part_size[2];
  std::vector<int> part_rotation_range;
  int num_part_rotation;
  int max_part_relative_rotation;
  int resolution_shift_limit;
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
  double scale_filter;
  double bin_size;
  double saturation;
  int num_stored_point;  /* number of stored points of lambda in exponential model */
  int numStoredPoint; /* deprecated*/
  int num_iteration;
  int numCandPart;
  int startx, starty, endx, endy;
  std::vector<int> part_loc_x;
  std::vector<int> part_loc_y;

  // second file
  MatCell_2<cv::Mat> allSelectedx;
  MatCell_2<cv::Mat> allSelectedy;
  MatCell_2<cv::Mat> allSelectedOrient;
  MatCell_1<cv::Mat> selectedlambda;
  MatCell_1<cv::Mat> selectedLogZ;
  MatCell_1<cv::Mat> allSymbol;
  MatCell_1<cv::Mat> allFilter;

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

bool GetCmdOptions(int argc, char** argv, SAOTConfig& config);
bool LoadConfigFile(const std::string& filename, SAOTConfig& config);

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_SAOT_CONFIG_HPP_
