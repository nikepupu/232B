#ifndef SAOT_SAOT_CONFIG_HPP_
#define SAOT_SAOT_CONFIG_HPP_

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
  int template_size[2], part_size[2];
  int *part_rotation_range;
  int num_part_rotation;
  int max_part_relative_rotation;
  double min_rotation_dif;
  int* rotation_range;
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
};

bool GetCmdOptions(int argc, char** argv, SAOTConfig& config);
bool LoadConfigFile(const std::string& filename, SAOTConfig& config);

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_SAOT_CONFIG_HPP_
