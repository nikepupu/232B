#ifndef SAOT_TEMPLATE_HPP_
#define SAOT_TEMPLATE_HPP_

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include "saot_config.hpp"
#include "saot_inference_Config.hpp"
#include "util/meta_type.hpp"
#include "exponential_model.hpp"

/**
 * \author Zilong Zheng
 */
namespace AOG_LIB {
namespace SAOT {

struct BasisParam {
  double selectedOrient, selectedx, selectedy, selectedlambda, selectedLogZ;
};

struct PartParam {
  float row, col, ori, scale;
};

/* \brief Affine transform of a list of components.
 * A component is a 4D point with its position, orientation and scale (length).
 */
void TemplateAffineTransform(const std::vector<PartParam>& in_comp,
                             std::vector<PartParam>& dest, float t_scale,
                             float r_scale, float c_scale, float rotation,
                             int n_ori);

class Template {
 public:
  // [Deprecated] for test only
  Template(const SAOTInferenceConfig& config) : config_(config) {}

  Template(const SAOTInferenceConfig& config, const MatCell_1<cv::Mat>& correlation,
           const MatCell_2<cv::Mat>& sum1_map_find,
           const MatCell_1<cv::Mat>& all_symbol,
           const MatCell_1<ExpParam>& exp_model);
  void Initialize();

  void RotateTemplate();

  void RotateS3Template();

  int num_cand_part;
  double part_loc_range;
  std::vector<int> part_loc_x0, part_loc_y0;
  std::vector<cv::Point2i> part_loc;

  // output variables for learning
  MatCell_1<std::vector<BasisParam> > selected_params_, larger_selected_params_;
  MatCell_2<std::vector<BasisParam> > all_selected_params_,
      larger_all_selected_params_;
  MatCell_1<cv::Mat> sum1_map_learn0_, deformed_template0_;
  MatCell_1<cv::Mat> common_template_, deformed_template_;
  std::vector<PartParam> S3_selected_parts_;
  std::vector<int> part_on_off_;
  MatCell_1<std::vector<PartParam> > all_S3_selected_parts_;

 private:
  SAOTInferenceConfig config_;
  cv::Size img_size_;
  MatCell_1<ExpParam> exp_model_;
  MatCell_1<cv::Mat> correlation_, all_symbol_;
  MatCell_2<cv::Mat> sum1_map_find_;
};

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_TEMPLATE_HPP_
