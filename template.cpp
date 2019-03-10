#include "template.hpp"
namespace AOG_LIB {
namespace SAOT {

void InitializeTemplate() {

}

void RotateTemplate() {

}

void RotateTemplateS3() {

}

/* \brief Affine transform of a list of components.
 * A component is a 4D point with its position, orientation and scale (length).
 */
void TemplateAffineTransform(const std::vector<Component> &in_comp,
                             MatCell_1<Component> &dest, float t_scale,
                             float r_scale, float c_scale, float rotation,
                             int n_ori) {
  cv::Mat A1(3, 3, CV_64F, 0.0), A2(3, 3, CV_64F, 0.0);
  int n_element = in_comp.size();
  A1.at<double>(0, 0) = c_scale * pow(2.0, t_scale / 2.0);
  A1.at<double>(1, 1) = r_scale * pow(2.0, t_scale / 2.0);

  float angle = rotation * CV_PI / n_ori;

  A2.at<double>(0, 0) = cos(angle);
  A2.at<double>(0, 1) = sin(angle);
  A2.at<double>(1, 0) = -sin(angle);
  A2.at<double>(1, 1) = cos(angle);

  cv::Vec3f pt, tmp;
  for(int i = 0; i < n_element; i++) {
      tmp[0] = in_comp[i].col;
      tmp[1] = -in_comp[i].row;
      tmp[2] = 1;

      
  }
}

} // namespace SAOT
} // namespace AOG_LIB
