#ifndef SAOT_Inference_HPP_
#define SAOT_Inference_HPP_

#include <string>
#include "saot_inference_Config.hpp"
#include "saot_inference_Config.hpp"
namespace AOG_LIB {
namespace SAOT{

	class SAOT_Inference 
	{

	public:
		SAOT_Inference();
		void TraceBack();

	private:
	void LoadConfig();
	SAOTInferenceConfig config;
	 
	std::string modelFolder;
	std::string configFile;
	std::string dotFile;
	std::string imageFolder;
	std::string imageName;
	std::string featureFolder;

	std::string featurefile;
	bool reLoadModel;
	int  it;
	bool doMorphBackS1map;
	bool doCropBackImage;
	bool showMatchedTemplate;
	bool showPartBoundingBox;
	bool showObjectBoundingBox;


	//////////// configuration file variables


	};
}
}



#endif