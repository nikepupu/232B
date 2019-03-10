#ifndef SAOT_TEMPLATE_HPP_
#define SAOT_TEMPLATE_HPP_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include "util/meta_type.hpp"
/**
 * \author Zilong Zheng
*/
namespace AOG_LIB {
namespace SAOT {

struct Component {
  int col, row, ori;
};

void InitializeTemplate();

void RotateTemplate();

void RotateTemplateS3();

// May use cvGetAffineTransform
void TemplateAffineTransform(const std::vector<Component> &in_comp,
                            MatCell_1<Component> &dest, float t_scale,
                            float r_scale, float c_scale, float rotation,
                            int n_ori);

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_TEMPLATE_HPP_
