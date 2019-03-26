#include "template.hpp"
#include "misc.hpp"
#include <cmath>
#include <boost/log/trivial.hpp>

namespace AOG_LIB {
namespace SAOT {

/* \brief Affine transform of a list of components.
 * A component is a 4D point with its position, orientation and scale (length).
 */
void TemplateAffineTransform(const std::vector<PartParam> &in_comp,
                             std::vector<PartParam> &dest_comp, float t_scale,
                             float r_scale, float c_scale, float rotation,
                             int n_ori) {
  if (dest_comp.size() > 0) {
    dest_comp.clear();
  }
  cv::Mat A1(3, 3, CV_64F, 0.0), A2(3, 3, CV_64F, 0.0);
  int n_element = in_comp.size();
  A1.at<double>(0, 0) = c_scale * pow(2.0, t_scale / 2.0);
  A1.at<double>(1, 1) = r_scale * pow(2.0, t_scale / 2.0);

  float angle = rotation * CV_PI / n_ori;

  A2.at<double>(0, 0) = cos(angle);
  A2.at<double>(0, 1) = sin(angle);
  A2.at<double>(1, 0) = -sin(angle);
  A2.at<double>(1, 1) = cos(angle);

  cv::Mat pt, tmp(1, 3, CV_64F);
  for (int i = 0; i < n_element; i++) {
    tmp.at<double>(0, 0) = in_comp[i].col;
    tmp.at<double>(0, 1) = -in_comp[i].row;
    tmp.at<double>(0, 2) = 1;
    pt = tmp * A1;
    tmp.at<double>(0, 0) = pt.at<double>(0, 0);
    tmp.at<double>(0, 1) = pt.at<double>(0, 1);
    tmp.at<double>(0, 2) = pt.at<double>(0, 2);
    pt = tmp * A2;

    PartParam out_param;
    out_param.col = round(pt.at<double>(0, 0));
    out_param.row = round(-pt.at<double>(0, 1));
    out_param.ori = in_comp[i].ori + rotation;
    out_param.scale = in_comp[i].scale + t_scale;
    dest_comp.push_back(out_param);
  }
}

Template::Template(const SAOTConfig &config,
                   const MatCell_1<cv::Mat> &correlation,
                   const MatCell_2<cv::Mat> &sum1_map_find,
                   const MatCell_1<cv::Mat> &all_symbol,
                   const MatCell_1<ExpParam> &exp_model)
    : config_(config), correlation_(correlation), sum1_map_find_(sum1_map_find),
      all_symbol_(all_symbol), exp_model_(exp_model) {
  for (int ind = 0; ind <= config_.template_size[0] - config_.part_size[0];
       ind += config.part_size[0]) {
    part_loc_x0.push_back(ind);
  }

  for (int ind = 0; ind <= config_.template_size[1] - config_.part_size[1];
       ind += config.part_size[1]) {
    part_loc_y0.push_back(ind);
  }

  num_cand_part = part_loc_x0.size() * part_loc_y0.size();
  for (int x : part_loc_x0) {
    for (int y : part_loc_y0) {
      part_loc.push_back(cv::Point2i(x, y));
    }
  }
}

void Template::Initialize() {
  // Prepare output variables for learning
  CreateMatCell1Dim<std::vector<BasisParam> >(selected_params_, num_cand_part);
  CreateMatCell1Dim<std::vector<BasisParam> >(larger_selected_params_,
                                             num_cand_part);
  CreateMatCell2Dim<std::vector<BasisParam> >(all_selected_params_,
                                             num_cand_part, 
                                             config_.num_part_rotation);
  CreateMatCell2Dim<std::vector<BasisParam> >(larger_all_selected_params_,
                                             num_cand_part, 
                                             config_.num_part_rotation);

  CreateMatCell1Dim<cv::Mat>(common_template_, num_cand_part);

  for (int i = 0; i < num_cand_part; i++) {
    common_template_[i] =
        cv::Mat::zeros(config_.part_size[0], config_.part_size[1], CV_64F);
  }

  CreateMatCell1Dim<cv::Mat>(deformed_template_, config_.num_image);
  for (int i = 0; i < config_.num_image; i++) {
    deformed_template_[i] =
        cv::Mat::zeros(config_.part_size[0], config_.part_size[1], CV_64F);
  }

  // Initialize learning from the starting image
  CreateMatCell2Dim<cv::Mat>(sum1_map_learn0_, 1, config_.num_orient);
  for (int orient = 0; orient < config_.num_orient; orient++) {
    sum1_map_learn0_[0][orient] = cv::Mat::zeros(
        config_.template_size[0], config_.template_size[1], CV_64F);
    // Ccopy(dest, src, startx, starty, 0, 0,
    // template_size, img_size, /*theta=*/0)
    // dest[x, y] = src[startx+x, starty+dy]
    int sizex = sum1_map_find_[config_.original_resolution-1][orient].rows;
    int sizey = sum1_map_find_[config_.original_resolution-1][orient].cols;
    int copy_size_x = fmin(sizex, config_.template_size[0]);
    int copy_size_y = fmin(sizey, config_.template_size[1]);
    sum1_map_learn0_[0][orient]
        .rowRange(0, copy_size_x)
        .colRange(0, copy_size_y) =
        sum1_map_find_[config_.original_resolution-1][orient]
            .rowRange(0, copy_size_x)
            .colRange(0, copy_size_y);
  }

  CreateMatCell1Dim<cv::Mat>(deformed_template0_, 1);
  deformed_template0_[0] =
      cv::Mat::zeros(config_.part_size[0], config_.part_size[1], CV_64F);

  MatCell_1<BasisParam> tmp_selected_params =
      CreateMatCell1Dim<BasisParam>(config_.num_element);
  cv::Mat tmp_common_template = cv::Mat::zeros(
      config_.template_size[0], config_.template_size[1], CV_64F);
      
  // SharedSketch(config_, cv::Size(config_.template_size[0], 
  //             config_.template_size[1]), sum1_map_learn0_, correlation_, 
  //             all_symbol_, tmp_common_template, deformed_template0_, 
  //             exp_model_, tmp_selected_params);

  // split the object template into non-overlapping partial templates
  for (int i_part = 0; i_part < num_cand_part; i_part++) {
    for (int ind = 0; ind < config_.num_element; ind++) {
      if (tmp_selected_params[ind].selectedx >= part_loc[i_part].x &&
          tmp_selected_params[ind].selectedx < config_.part_size[0] &&
          tmp_selected_params[ind].selectedy >= part_loc[i_part].y &&
          tmp_selected_params[ind].selectedy < config_.part_size[1]) {
        BasisParam selected_param;
        selected_param.selectedOrient = tmp_selected_params[ind].selectedOrient;
        selected_param.selectedx =
            tmp_selected_params[ind].selectedx - part_loc[i_part].x;
        selected_param.selectedy =
            tmp_selected_params[ind].selectedy - part_loc[i_part].y;
        selected_param.selectedlambda = tmp_selected_params[ind].selectedlambda;
        selected_param.selectedLogZ = tmp_selected_params[ind].selectedLogZ;
        selected_params_[i_part].push_back(selected_param);
        // add a small margin to make sure all Gabor elements are displayed
        // fully
      }
    }
  }

  // split the object template into larger overlapping partial templates
  // (context sensitive)
  for (int i_part = 0; i_part < num_cand_part; i_part++) {
    for (int ind = 0; ind < config_.num_element; ind++) {
      if (tmp_selected_params[ind].selectedx >= part_loc[i_part].x &&
          tmp_selected_params[ind].selectedx < config_.part_size[0] &&
          tmp_selected_params[ind].selectedy >= part_loc[i_part].y &&
          tmp_selected_params[ind].selectedy < config_.part_size[1]) {
        BasisParam selected_param;
        selected_param.selectedOrient = tmp_selected_params[ind].selectedOrient;
        selected_param.selectedx = tmp_selected_params[ind].selectedx -
                                   part_loc[i_part].x + config_.part_margin[0];
        selected_param.selectedy = tmp_selected_params[ind].selectedy -
                                   part_loc[i_part].y + config_.part_margin[1];
        selected_param.selectedlambda = tmp_selected_params[ind].selectedlambda;
        selected_param.selectedLogZ = tmp_selected_params[ind].selectedLogZ;
        larger_selected_params_[i_part].push_back(selected_param);
        // add a small margin to make sure all Gabor elements are displayed
        // fully
      }
    }
  }

  RotateTemplate();

  // Init S3 template
  for (int i = 0; i < num_cand_part; i++) {
    part_on_off_.push_back(1);
    PartParam param;
    param.row =
        part_loc[i].x + static_cast<int>(floor(config_.part_size[0] / 2));
    param.col =
        part_loc[i].y + static_cast<int>(floor(config_.part_size[0] / 2));
    param.ori = 0;
    S3_selected_parts_.push_back(param);
  }

  CreateMatCell1Dim<std::vector<PartParam> >(all_S3_selected_parts_,
                                            config_.num_part_rotation);

  for (int i = 0; i < config_.num_rotate; i++) {
    std::vector<PartParam> params;
    for (int j = 0; j < num_cand_part; j++) {
      PartParam param = {0, 0, 0};
      params.push_back(param);
    }
    all_S3_selected_parts_[i] = params;
  }

  RotateS3Template();
}

void Template::RotateTemplate() {
  // const int dense_x = static_cast<int>(-floor(config_.part_size[0] / 2));
  // const int dense_y = static_cast<int>(-floor(config_.part_size[1] / 2));
  // int count = 0;
  // MatCell_1<PartParam> in_part =
  //     CreateMatCell1Dim<PartParam>(config_.part_size[0] *
  //     config_.part_size[1]);

  // for (int y = 0; y < config_.part_size[1]; y++) {
  //   for (int x = 0; x < config_.part_size[0]; x++) {
  //     in_part[count].row = dense_x + x;
  //     in_part[count].col = dense_y + y;
  //     in_part[count].ori = 0;
  //     count += 1;
  //   }
  // }

  // non-overlapping partial templates
  for (int i_part = 0; i_part < num_cand_part; i_part++) {
    double center_x = floor(config_.part_size[0] / 2);
    double center_y = floor(config_.part_size[1] / 2);

    for (int i = 0; i < selected_params_[i_part].size(); i++) {
      selected_params_[i_part][i].selectedx -= center_x;
      selected_params_[i_part][i].selectedy -= center_y;
    }

    for (int r = 0; r < config_.num_part_rotation; r++) {
      int rot = config_.part_rotation_range[r];
      float t_scale = 0, r_scale = 1, c_scale = 1;
      std::vector<PartParam> dest_part;
      std::vector<PartParam> in_part;
      for (auto param : selected_params_[i_part]) {
        PartParam part;
        part.row = param.selectedx;
        part.col = param.selectedy;
        part.ori = param.selectedOrient;
        in_part.push_back(part);
      }
      TemplateAffineTransform(in_part, dest_part, t_scale, r_scale, c_scale,
                              rot, config_.num_orient);
      for (auto part : dest_part) {
        BasisParam param;
        param.selectedx = part.row;
        param.selectedy = part.col;
        if (part.ori < 0) {
          part.ori += config_.num_orient;
        } else if (part.ori >= config_.num_orient) {
          part.ori -= config_.num_orient;
        }
        param.selectedOrient = part.ori;
        all_selected_params_[i_part][r].push_back(param);
      }
    }
  }

  // overlapping partial templates
  for (int i_part = 0; i_part < num_cand_part; i_part++) {
    double center_x = floor(config_.part_size[0] / 2 + config_.part_margin[0]);
    double center_y = floor(config_.part_size[1] / 2 + config_.part_margin[1]);
    for (int i = 0; i < larger_selected_params_[i_part].size(); i++) {
      larger_selected_params_[i_part][i].selectedx -= center_x;
      larger_selected_params_[i_part][i].selectedy -= center_y;
    }

    for (int r = 0; r < config_.num_part_rotation; r++) {
      int rot = config_.part_rotation_range[r];
      float t_scale = 0, r_scale = 1, c_scale = 1;
      std::vector<PartParam> dest_part;
      std::vector<PartParam> in_part;
      for (auto param : larger_selected_params_[i_part]) {
        PartParam part;
        part.row = param.selectedx;
        part.col = param.selectedy;
        part.ori = param.selectedOrient;
        in_part.push_back(part);
      }
      TemplateAffineTransform(in_part, dest_part, t_scale, r_scale, c_scale,
                              rot, config_.num_orient);
      for (auto part : dest_part) {
        BasisParam param;
        param.selectedx = part.row;
        param.selectedy = part.col;
        if (part.ori < 0) {
          part.ori += config_.num_orient;
        } else if (part.ori >= config_.num_orient) {
          part.ori -= config_.num_orient;
        }
        param.selectedOrient = part.ori;
        larger_all_selected_params_[i_part][r].push_back(param);
      }
    }
  }
}

void Template::RotateS3Template() {
  double center_x = floor(config_.part_size[0] / 2);
  double center_y = floor(config_.part_size[1] / 2);

  for (int i = 0; i < num_cand_part; i++) {
    S3_selected_parts_[i].row -= center_x;
    S3_selected_parts_[i].col -= center_y;
  }

  for (int r = 0; r < config_.num_part_rotation; r++) {
    int rot = config_.part_rotation_range[r];
    float t_scale = 0, r_scale = 1, c_scale = 1;
    std::vector<PartParam> dest_part;
    TemplateAffineTransform(S3_selected_parts_, dest_part, t_scale, r_scale,
                            c_scale, rot, config_.num_orient);
    all_S3_selected_parts_[r] = dest_part;
  }
}

} // namespace SAOT
} // namespace AOG_LIB