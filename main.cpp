#include "detector.hpp"
#include "saot_inference_Config.hpp"
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#define BOOST_LOG_DYN_LINK 1

using  AOG_LIB::SAOT::SAOTInferenceConfig;
using AOG_LIB::SAOT::SAOT_Inference; 

int main(int argc, char ** argv)
{

	//SAOT_Inference *temp = new SAOT_Inference();
	//temp->LoadConfig();
	

	boost::shared_ptr<AOG_LIB::SAOT::SAOT_Inference> stInference = boost::make_shared<AOG_LIB::SAOT::SAOT_Inference>();

	return 0;
}
