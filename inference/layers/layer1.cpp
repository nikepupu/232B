#include "layer.hpp"
#include "../saot_config.hpp"
#include "../misc.hpp"
#include "../filter.hpp"

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/multi_array.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace AOG_LIB {
namespace SAOT {

void ComputeMAX1(const SAOTConfig &config, const MatCell_2<cv::Mat> &map_sum1_find,
                 MatCell_2<cv::Mat> &map_max1, MatCell_2<cv::Mat> &m1_trace,
                 cv::Mat &m1_shift_row, cv::Mat &m1_shift_col, cv::Mat &m1_shift_ori)
{

  // usage : [MAX1map(iRes,:) M1Trace(iRes,:) M1RowShift M1Co lShift M1Or iShifted] = 
  // mexc_ComputeMAX1(numOrient, SUM1mapFind(iRes,:), locationShiftLimit, m1_shift_oriLimit, 1);

  /////////////////////////// PART1 : variable declaration ////////////////////////

  int num_orient = config.num_orient;                     /* number of orientations for Gabor elements */
  int location_shift_limit = config.location_shift_limit; /* allowed location perturbation, relative to the Gabor filter size */
  int orient_shift_limit = config.orient_shift_limit;     /* allowed orientation perturbation */
  int subsample = config.subsample;                       /* sub sampling step length */
  int nGaborScale = (int)(map_sum1_find.shape()[1] / num_orient);   /* number of Gabor scales */   
  int numShift = (location_shift_limit * 2 + 1) * (orient_shift_limit * 2 + 1);
                                                          /* number of local shifts, the same for all orientations and all scales */ 

  int orient, i, j, shift, ci, cj, si, sj, os, iS, x, y, x1, y1, orient1, mshift = 0;
  float alpha, maxResponse, r;
  const float **S1Map; /* SUM1 maps */

  int num_image = map_sum1_find.shape()[0];
  int num_res = map_sum1_find.shape()[1];

  ///////////////////////////////// PART2 : StoreShift /////////////////////////////////

  for (iS = 0; iS < nGaborScale; ++iS) /* for each scale */
    for (orient = 0; orient < num_orient; orient++)
    { /* for each orientation */

      /* order the shifts from small to large */
      alpha = (float)3.1415926 * orient / num_orient;
      ci = 0;
      for (i = 0; i <= location_shift_limit; ++i) // location shifts
        for (si = 0; si <= 1; si++)
        { /* sign of i */
          double location_shift = i;
          cj = 0;
          for (j = 0; j <= orient_shift_limit; j++) // orientation shifts
            for (sj = 0; sj <= 1; sj++)
            {                                                 /* sign of j */
              shift = ci * (2 * orient_shift_limit + 1) + cj; /* hybrid counter for location and orientation */
              m1_shift_row.at<int>(iS * num_orient + orient, shift) = (int)floor(.5 + location_shift * (si * 2 - 1) * cos(alpha));
              m1_shift_col.at<int>(iS * num_orient + orient, shift) = (int)floor(.5 + location_shift * (si * 2 - 1) * sin(alpha));
              os = orient + j * (sj * 2 - 1);
              if (os < 0)
                os += num_orient;
              else if (os >= num_orient)
                os -= num_orient;
              m1_shift_ori.at<int>(iS * num_orient + orient, shift) = os;
              if (j > 0 || sj == 1)
                cj++; /* triggers the orientation counter */
            }
          if (i > 0 || si == 1)
            ci++;
        }
    }

  ///////////////////////////////// PART3 : Compute /////////////////////////////////

  /* local maximum pooling at (x, y, orient) */

  for (int iRes = 0; iRes < num_res; iRes++){
    
    int height = map_sum1_find[iRes][0].rows;
    int width = map_sum1_find[iRes][0].cols;

    int sizexSubsample = (int)floor((double)height / subsample);
    int sizeySubsample = (int)floor((double)width / subsample);

    for (iS = 0; iS < nGaborScale; ++iS)
      for (orient = 0; orient < num_orient; orient++)

        for (y = 0; y < sizeySubsample; y++) for (x = 0; x < sizexSubsample; x++) {
          maxResponse = -1e10;
          mshift = 0;
          for (shift = 0; shift < numShift; shift++) {
            x1 = x * subsample + m1_shift_row.at<int>(iS * num_orient + orient, shift);
            y1 = y * subsample + m1_shift_col.at<int>(iS * num_orient + orient, shift);
            orient1 = m1_shift_ori.at<int>(iS * num_orient + orient, shift);
            if ((x1 >= 0) && (x1 < height) && (y1 >= 0) && (y1 < width)) {
              r = map_sum1_find[iRes][iS * num_orient + orient1].at<int>(x1, y1);
              if (r > maxResponse) {
                maxResponse = r;
                mshift = shift;
              }
            }
          }
          map_max1[iRes][iS * num_orient + orient].at<int>(x, y) = maxResponse;
          m1_trace[iRes][iS * num_orient + orient].at<int>(x, y) = (float)mshift;
        }
  }
}

void ComputeSUM1(const SAOTConfig& config, const MatCell_1<cv::Mat> &images, 
    const MatCell_1<cv::Mat> &all_filter, MatCell_2<cv::Mat> &map_sum1_find)
{

  int num_resolution = config.num_resolution;
  int num_image = images.shape()[0];
  //BOOST_LOG_TRIVIAL(info) << "start filtering training images at all resolutions";
  for (int i = 0; i < num_image; i++) {
    const cv::Mat& image = images[i];
    
    MatCell_1<cv::Mat> multi_resolution_images =
                  CreateMatCell1Dim<cv::Mat>(num_resolution);
    for (int j = 0; j < num_resolution; j++) {
      double resolution = 0.6 + j * 0.2;
      cv::Mat dest_img;
      cv::Size dsize(image.rows * resolution, image.cols * resolution);
      cv::resize(image, dest_img, dsize, cv::INTER_NEAREST);
      multi_resolution_images[j] = dest_img;
    }
  
    ApplyFilterfft(config, multi_resolution_images, all_filter, map_sum1_find);
    Sigmoid(config, map_sum1_find);
  }
}

}  // namespace SAOT
}  // namespace AOG_LIB
