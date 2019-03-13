#ifndef SAOT_LAYERS_LAYER_HPP_
#define SAOT_LAYERS_LAYER_HPP_
#include "../util/meta_type.hpp"
#include "../saot_config.hpp"


#include <string>
#include <vector>

/**
 * \author Yifei Xu
*/
namespace AOG_LIB {
namespace SAOT {

void ComputeMAX1(const SAOTConfig& config, const MatCell_2<cv::Mat> &map_sum1_find, 
                 MatCell_2<cv::Mat> &map_max1, MatCell_2<cv::Mat> &m1_trace,
                 cv::Mat &m1_shift_row, cv::Mat &m1_shift_col, cv::Mat &m1_shift_ori);
// usage : [MAX1map(iRes,:) M1Trace(iRes,:) M1RowShift M1ColShift M1OriShifted] =
// mexc_ComputeMAX1(numOrient, SUM1mapFind(iRes,:), locationShiftLimit, orientShiftLimit, 1);
// MatCell_2 : num_resulotion * num_orient

void ComputeSUM1(const SAOTConfig& config, const MatCell_1<cv::Mat> &images, 
                const MatCell_1<cv::Mat> &all_filter, MatCell_2<cv::Mat> &map_sum1_find);

// Usage : Used before ComputeMax1 in order to load sum1 map.
// Inside this func, we call : ApplyFilterfft(config, multi_resolution_images, allFilter, localHalfx, localHalfy, thresholdFactor); Do not know what is this.

void FakeMAX2(const SAOTConfig& config, const MatCell_3<cv::Mat> &map_sum2, 
    const MatCell_3<cv::Mat> &m2_location_trace, const MatCell_3<cv::Mat> &m2_transform_trace, 
    MatCell_3<cv::Mat> &map_fake_max2);

// usage :  tmpMAX2map = mexc_FakeMAX2( SUM2map, largerMAX2LocTrace, largerMAX2TransformTrace,
// templateAffMat, int32(ones(numel(SUM2map),1) * sqrt(partSizeX*partSizeY)/subsampleS2), M2RowColShift );

void ComputeSUM2(const SAOTConfig &config, const MatCell_2<cv::Mat> &map_max1,
                 const MatCell_3<template_filter> &template_layer2, MatCell_3<cv::Mat> &map_sum2);

// usage :  tmpS2 = mexc_ComputeSUM2( numOrient, MAX1map(iRes,:), S2T(:), subsampleS2 ); SUM2map(:,:,iRes) = reshape(tmpS2,[numPartRotate numCandPart]);
// This function direct output SUM2map (stored in map_sum2)
// MatCell_3 : num_part_orient * num_parts * num_resulotion

void ComputeMAX2(const SAOTConfig &config, const MatCell_3<cv::Mat> &map_sum2,
                 MatCell_3<cv::Mat> &map_max2, MatCell_3<cv::Mat> &m2_location_trace, MatCell_3<cv::Mat> &m2_transform_trace, MatCell_1<double> &rowShift, MatCell_1<double> &colShift);

// usage :  [tmpMAX2 tmpMAX2LocTrace tmpMAX2TransformTrace M2RowColShift] = mexc_ComputeMAX2( templateAffMat(:), ...
// largerSUM2map(:,:,iRes), locationPerturbFraction, int32(sqrt(partSizeX*partSizeY)*ones(numPartRotate*numCandPart,1)/subsampleS2), subsampleM2);
// This function direct output largeMAX2map and other two.
// MatCell_2 &templateAffMat : num_part_orient * num_parts, do not know what is this.
// MatCell_3<cv::Mat> &map_max2 : num_resolution * num_part_rotate * num_cand_part (differ to Matlab, resolution become dim 0)

void ComputeSUM3(const SAOTConfig &config, const MatCell_3<cv::Mat> &map_max2,
                 const MatCell_2<template_filter> &template_layer3, MatCell_2<cv::Mat> &map_sum3);
// usage :  SUM3map(iRes) = mexc_ComputeSUM3( tmpM2(:), S3T(r), 1, numPartRotate );
// Differ to the definition in matlab. This function direct output all SUM3Map include all rotation, resolution and location.
// MatCell_1 &template_layer3 is a list of select template, to be defined by Tengyu Liu.
// MatCell_2 &map_sum3 : num_resolution * subsample_S3(typically=1)

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_LAYERS_LAYER_HPP_
