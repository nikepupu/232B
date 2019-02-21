#ifndef SAOT_SAOT_INFERENCE_CONFIG_HPP_
#define SAOT_SAOT_INFERENCE_CONFIG_HPP_

#include <string>
#include <vector>

namespace AOG_LIB {
namespace SAOT{

struct SAOTInferenceConfig {
//from the first file
int subsampleS2;
int numOrient;
double ***allFilter;
double ***allSymbol;
int* partRotationRange;
double locationPerturbFraction;
int numPartRotate;

int maxPartRelativeRotation;
int resolutionShiftLimit;
int numCandPart;
int* PartLocX;
int* PartLocY;

int sizeTemplatex;
int sizeTemplatey;
int partSizeX;
int partSizeY;
double minRotationDif;
int *rotationRange;
int numResolution;
int resizeFactor;
int numElement;

int localHalfx;
int localHalfy;
double thresholdFactor;
int saturation;
int locationShiftLimit;
int orientShiftLimit;

// the second file
std::vector<std::vector<int> > allSelectedx;
std::vector<std::vector<int> > allSelectedy;
std::vector<std::vector<int> > allSelectedOrient;
std::vector<std::vector<double> > selectedlambda;
std::vector<std::vector<double> > selectedLogZ;

std::vector<std::vector<int> > largerAllSelectedx;
std::vector<std::vector<int> > largerAllSelectedy;
std::vector<std::vector<int> > largerAllSelectedOrient;
std::vector<std::vector<double> > largerSelectedlambda;
std::vector<std::vector<double> > largerSelectedLogZ;

//thrid file
int *PartOnOff;
std::vector<std::vector<int> > allS3SelectedRow;
std::vector<std::vector<int> > allS3SelectedCol;
std::vector<std::vector<int> > allS3SelectedOri;

std::vector<int> selectedPart;



};

bool LoadSAOTInferenceConfigFile(const std::string &filename1,  const std::string &filename2, const std::string &filename3, SAOTInferenceConfig &config);

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_SAOT_CONFIG_HPP_
