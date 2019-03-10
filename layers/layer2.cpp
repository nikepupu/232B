#include "../saot_config.hpp"
#include "layer.hpp"

#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace AOG_LIB {
namespace SAOT {

void ComputeSUM2(const SAOTConfig &config, const MatCell_2<cv::Mat> &map_max1,
                 const MatCell_3<template_filter> &template_layer2, MatCell_3<cv::Mat> &map_sum2) {

    int subsample = config.subsample_S2;
    int num_res = map_max1.shape()[0];
    int num_part_rotate = template_layer2.shape()[0];
    int numCandPart = template_layer2.shape()[1];
    int nElement = template_layer2.shape()[2];  /* number of elements for each template */
    int subsampleS2 = config.subsample_S2;      /* step length when scanning the SUM2 template over MAX1 map */

    for (int iRes = 0; iRes < map_max1.shape()[0]; iRes++) {
        int heightM1Map = map_max1[iRes][0].rows;
        int widthM1Map = map_max1[iRes][0].cols;
        int heightS2Map = (int)floor((double)heightM1Map / (double)subsampleS2);
        int widthS2Map = (int)floor((double)widthM1Map / (double)subsampleS2);
        for (int i = 0; i < num_part_rotate; i++) for (int j = 0; j < numCandPart; j++)
        for (int k = 0; k < nElement; k++) {
            template_filter t = template_layer2[i][j][k];
            int iOriM1 = (int)t.ori;
            int iScaleM1 = (int)t.scale;
            for (int iColS2 = 0; iColS2 < widthS2Map; ++iColS2) {
                int iColM1 = (int)floor(.5 + (iColS2 + .5) * subsampleS2 + t.col);
                for (int iRowS2 = 0; iRowS2 < heightS2Map; ++iRowS2) {
                    int iRowM1 = (int)floor(.5 + (iRowS2 + .5) * subsampleS2 + t.row);
                    if (iRowM1 < 0 || iRowM1 >= heightM1Map || iColM1 < 0 || iColM1 >= widthM1Map) {
                        map_sum2[iRes][i][j].at<int>(iRowS2, iColS2) += -t.logZ + t.lambda * 0;
                        continue;
                    }
                    map_sum2[iRes][i][j].at<int>(iRowS2, iColS2) +=
                        -t.logZ + t.lambda * map_max1[iRes][iOriM1 + iScaleM1 * config.num_orient].at<int>(iRowM1, iColM1);
                }
            }
        }
    }
}

void ComputeMAX2(const SAOTConfig &config, const MatCell_3<cv::Mat> &map_sum2, 
MatCell_3<cv::Mat> &map_max2, MatCell_3<cv::Mat> &m2_location_trace, MatCell_3<cv::Mat> &m2_transform_trace)
{

    /////////////////////////// PART1 : variable declaration ////////////////////////

    double location_perturb_fraction = config.location_perturb_fraction;
    int subsample = config.subsample_S2;
    int height, width;                      /* size of MAX1 maps */
    int sizexsubsampleM2, sizeysubsampleM2; /* (down sampled) size of M1 map */
    int subsampleM2;                        /* sub sampling step length */

    ///////////////////////////////// PART2 : StoreShift /////////////////////////////////

    int orient, i, j, shift, si, sj, os, iS, x, y, x1, y1, orient1, mshift;
    float dc, dr, alpha, stepsize, maxResponse, r;
    int iT, jT, jjT, bestLocationshift, bestTemplate;
    bool success;
    float *map_max2_toUpdate, *map_max2_neighbor;

    /* store all the possible shifts for all templates */
    stepsize = 0.05;

    /* For each template, the number of location shifts is the same */
    /* We store the shifts as relative positions to the template size. */

    int sizeTemplate = (int)(sqrt(config.part_size[0]*config.part_size[1]) / config.subsample_S2);
    int nLocationShift = (int)floor((location_perturb_fraction * 2) / stepsize + 1);
    nLocationShift *= nLocationShift;
    MatCell_1<double> rowShift(boost::extents[nLocationShift]);
    MatCell_1<double> colShift(boost::extents[nLocationShift]);
    j = 0;
    for (dc = -location_perturb_fraction; dc <= location_perturb_fraction; dc += stepsize)
        for (dr = -location_perturb_fraction; dr <= location_perturb_fraction; dr += stepsize) {
            rowShift[j] = dr;
            colShift[j] = dc;
            ++j;
        }

    ///////////////////////////////// PART3 : Compute /////////////////////////////////

    /* For each channel, perform local maximization on its own. */
    for (int iRes = 0; iRes < map_max2.shape()[0]; iRes++) {

        height = map_sum2[iRes][0][0].rows;
        width = map_sum2[iRes][0][0].cols;
        sizexsubsampleM2 = (int)floor((double)height / subsampleM2);
        sizeysubsampleM2 = (int)floor((double)width / subsampleM2);
        for (int i = 0; i < config.num_part_rotation; i++)
            for (int j = 0; j < config.numCandPart; j++)
                for (x = 0; x < sizexsubsampleM2; x++)
                    for (y = 0; y < sizeysubsampleM2; y++) {
                        success = false;
                        maxResponse = -1e10;

                        for (shift = 0; shift < nLocationShift; ++shift) {
                            x1 = (int)floor((x)*subsampleM2 + rowShift[shift] * sizeTemplate);
                            y1 = (int)floor((y)*subsampleM2 + colShift[shift] * sizeTemplate);
                            if ((x1 >= 0) && (x1 < height) && (y1 >= 0) && (y1 < width)) {
                                r = map_sum2[iRes][i][j].at<int>(x1, y1);
                                if (r > maxResponse) {
                                    success = true;
                                    maxResponse = r;
                                    bestLocationshift = shift;
                                    bestTemplate = i * config.numCandPart + j;
                                }
                            }
                        }

                        map_max2[iRes][i][j].at<int>(x, y) = maxResponse;
                        m2_location_trace[iRes][i][j].at<int>(x, y) = bestLocationshift;
                        m2_transform_trace[iRes][i][j].at<int>(x, y) = bestTemplate;
                    }

        /* Then, combine the channels. */
        for (int i = 0; i < config.num_part_rotation; i++)
            for (int j = 0; j < config.numCandPart; j++)
            {
                float angle1 = PI / config.num_orient * config.part_rotation_range[i];
                for (int i2 = 0; i2 < config.num_part_rotation; i2++) {
                    float angle2 = PI / config.num_orient * config.part_rotation_range[i2];
                    if (pow(sin(angle1) - sin(angle2), 2) + pow(cos(angle1) - cos(angle2), 2) <= config.min_rotation_dif)
                        for (x = 0; x < sizexsubsampleM2; x++)
                            for (y = 0; y < sizeysubsampleM2; y++)
                                if (map_max2[iRes][i2][j].at<int>(x, y) > map_max2[iRes][i][j].at<int>(x, y))
                                {
                                    map_max2[iRes][i][j].at<int>(x, y) = map_max2[iRes][i2][j].at<int>(x, y);
                                    m2_transform_trace[iRes][i][j].at<int>(x, y) = m2_transform_trace[iRes][i2][j].at<int>(x, y);
                                    m2_location_trace[iRes][i][j].at<int>(x, y) = m2_location_trace[iRes][i2][j].at<int>(x, y);
                                }
                }
            }
    }
}

void FakeMAX2(const SAOTConfig& config, const MatCell_3<cv::Mat> &map_sum2, 
    const MatCell_3<cv::Mat> &m2_location_trace, const MatCell_3<cv::Mat> &m2_transform_trace, 
    MatCell_3<cv::Mat> &map_fake_max2) {

    float stepsize = 0.05;
    int location_perturb_fraction = config.location_perturb_fraction;
    int sizeTemplate = (int)(sqrt(config.part_size[0] * config.part_size[1]) / config.subsample_S2);
    int nLocationShift = (int)floor((config.location_perturb_fraction * 2) / stepsize + 1);
    nLocationShift *= nLocationShift;
    MatCell_1<double> rowShift(boost::extents[nLocationShift]);
    MatCell_1<double> colShift(boost::extents[nLocationShift]);
    int j = 0;
    for (int dc = -location_perturb_fraction; dc <= location_perturb_fraction; dc += stepsize)
        for (int dr = -location_perturb_fraction; dr <= location_perturb_fraction; dr += stepsize) {
            rowShift[j] = dr;
            colShift[j] = dc;
            ++j;
        }

    for (int iRes = 0; iRes < map_sum2.shape()[0]; ++iRes) {
        int height = map_sum2[iRes][0][0].rows;
        int width = map_sum2[iRes][0][0].cols;
        for (int iRot = 0; iRot < map_sum2.shape()[1]; ++iRot)
            for (int iPart = 0; iPart < map_sum2.shape()[2]; ++iPart)
                for (int x = 0; x < height; ++x) 
                    for (int y = 0; y < width; ++y)
                    {
                        // use TransformTrace iPart and iRot to find the rotation
                        int idx = m2_transform_trace[iRes][iPart][iRot].at<int>(x, y); // the index for the transformed template (hybrid of iRot and iPart)
                        int s_rot = (int)(idx / config.numCandPart);
                        int s_part = idx % config.numCandPart;
                        // use x, y and iRes to find the translation
                        int indTranslation = m2_location_trace[iRes][iPart][iRot].at<int>(x, y);
                        int xx = (int)floor(x + rowShift[indTranslation] * (float)sizeTemplate);
                        int yy = (int)floor(y + colShift[indTranslation] * (float)sizeTemplate);

                        // fetch the point in S2Map
                        map_fake_max2[iRes][iPart][iRot].at<int>(x, y) = map_sum2[iRes][s_rot][s_part].at<int>(xx, yy);
                    }
    }
}

}  // namespace SAOT
}  // namespace AOG_LIB
