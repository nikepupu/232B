#include "detector.hpp"
#include "saot_config.hpp"
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#ifndef BOOST_LOG_DYN_LINK
#define BOOST_LOG_DYN_LINK 1
#endif

using AOG_LIB::SAOT::SAOT_Inference; 
using AOG_LIB::SAOT::SAOTConfig;

int main(int argc, char ** argv)
{

	//SAOT_Inference *temp = new SAOT_Inference();
	//temp->LoadConfig();
	

	boost::shared_ptr<AOG_LIB::SAOT::SAOT_Inference> stInference = boost::make_shared<AOG_LIB::SAOT::SAOT_Inference>();
	

	return 0;
}
