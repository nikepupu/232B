#include "layer.hpp"

#include "../saot_config.hpp"
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>


namespace AOG_LIB {
namespace SAOT {

void ComputeSUM3(const SAOTConfig& config, const MatCell_3<cv::Mat> &map_max2, 
    const MatCell_2<template_filter> &template_layer3, MatCell_2<cv::Mat> &map_sum3) {

    int nS3Template = template_layer3.shape()[0];               /* number of S3 templates */
    int nElement = template_layer3.shape()[1];                  /* number of elements for each template */
    int subsampleS3 = config.subsample_S3; /* step length when scanning the SUM2 template over MAX1 map */
    int i, j, k, iRowS3, iColS3, iRowM2, iColM2, jTS2, jTS2Trans;
    double r;
    int rowOffset = 0;
    int colOffset = 0;

    /*
     * compute the lowest value in each M2 map: to deal with out-of-boundary case
     */

    MatCell_3<double> minValPerM2Map(boost::extents[map_max2.shape()[0]][map_max2.shape()[1]][map_max2.shape()[2]]);
    for (i = 0; i < map_max2.shape()[0]; i++)
    for (j = 0; j < map_max2.shape()[1]; j++)
    for (k = 0; k < map_max2.shape()[2]; k++) {
        cv::minMaxLoc(map_max2[i][j][k], &minValPerM2Map[i][j][k], &r);
    }

    /* About the visiting order in the FOR loop:
     *      The scan over M1 map positions should be inner-most.
     */
    for (int iRes = 0; i < iRes < map_max2.shape()[2]; iRes++) {
        int heightM2Map = map_max2[iRes][0][0].rows;
        int widthM2Map = map_max2[iRes][0][0].cols;
        int heightS3Map = (int)floor((double)heightM2Map / subsampleS3);
        int widthS3Map = (int)floor((double)widthM2Map / subsampleS3);
        for (i = 0; i < nS3Template; i++) for (j = 0; j < nElement; j++) {
            template_filter t = template_layer3[i][j];
            jTS2 = t.ind;
            jTS2Trans = t.trans;
            for (iColS3 = 0; iColS3 < widthS3Map; ++iColS3) {
                iColM2 = (int)(colOffset + iColS3 * subsampleS3 + t.col);
                for (iRowS3 = 0; iRowS3 < heightS3Map; ++iRowS3) {
                    iRowM2 = (int)(rowOffset + iRowS3 * subsampleS3 + t.row);
                    if (iRowM2 < 0 || iRowM2 >= heightM2Map || iColM2 < 0 || iColM2 >= widthM2Map) {
                        map_sum3[iRes][i].at<int>(iRowS3, iColS3) +=
                            minValPerM2Map[iRes][jTS2Trans][jTS2] * t.lambda - t.logZ;
                        continue;
                    }
                    map_sum3[iRes][i].at<int>(iRowS3, iColS3) +=
                        -t.logZ + t.lambda * map_max2[iRes][jTS2Trans][jTS2].at<int>(iRowM2, iColM2);
                }
            }
        }
    }
    
}

}  // namespace SAOT
}  // namespace AOG_LIB
