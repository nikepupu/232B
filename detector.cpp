// ToDo:
// implement: size(a,num), size(a) , min
#include "detector.hpp"
#include "saot_inference_Config.hpp"

namespace AOG_LIB {
namespace SAOT {

SAOT_Inference::SAOT_Inference()
{


	modelFolder = "working";
	configFile = "Config.mat";
	dotFile = "inferenceResult.dot";
	imageFolder = "positiveImage";
	imageName = "l1.jpg";
	featureFolder = "working";

	reLoadModel = false;
	it = 25;
	doMorphBackS1map = true;
	doCropBackImage = true;
	showMatchedTemplate = true;
	showPartBoundingBox = true;
	showObjectBoundingBox = true;

	featurefile   = featureFolder+"/"+imageName+".mat";
	LoadConfig();
}

void SAOT_Inference::LoadConfig()
{
	AOG_LIB::SAOT::LoadSAOTInferenceConfigFile( configFile, "./"+modelFolder+"/partModel_iter"+std::to_string(it)+".mat" , 
		"./"+modelFolder+"/objectModel_iter"+std::to_string(it)+".mat" ,config);
}

//void AOG_Inference::TraceBack()
//{
	// int gaborCount = 0;
	// for(int iPart = 1; iPart <= numCandPart; i++)
	// {
	// 	r = find( allS3SelectedOri[bestRotInd,iPart] == partRotationRange ); // the index of part rotation
	// 	Fx = therex + floor(.5+allS3SelectedRow[bestRotInd][iPart]/subsampleM2/subsampleS2);
	// 	Fy = therey + floor(.5+allS3SelectedCol[bestRotInd][iPart]/subsampleM2/subsampleS2); // sub-sampled position

	// 	// size need to return a 2d array
	// 	int *imagesize = size(MAX2map[r][iPart][bestRes]); // subsampled image size 
		
	// 	// set default values of some output variables
	// 	bestPartRes = bestRes;
	// 	partScores = min(MAX2map[r][iPart][bestRes];
		
	// 	if（Fx >= 1 && Fx <= imagesize[0] && Fy >= 1 && Fy <= imagesize[1]）
	// 		tmp = MAX2map[r][iPart][bestRes];
	// 		partScores[iPart] = tmp[Fx][Fy];
			
	// 		tmp = MAX2ResolutionTrace[r][iPart][bestRes];
	// 		bestPartRes = tmp[Fx][Fy] + 1; % best part resolution
	// 		// current_size is a 2d array, size need to return a pointer
	// 		current_size = size(tmp);
			
	// 		tmp = largerMAX2LocTrace[r][iPart][bestPartRes];
	// 		new_size = size(tmp);
	// 		Fx = floor(.5+Fx*double(new_size)/current_size);
	// 		Fy = floor(.5+Fy*double(new_size)/current_size);
			
	// 		if(Fx >= 1 && Fx <= size(tmp,1) && Fy >= 1 && Fy <= size(tmp,2))
	// 			translationInd = tmp[Fx][Fy] + 1;
	// 		else
	// 			translationInd = floor(size(M2RowColShift,1)/2);
			
			
	// 		tmp = largerMAX2TransformTrace[r][iPart][bestPartRes]
	// 		if Fx >= 1 && Fx <= size(tmp,1) && Fy >= 1 && Fy <= size(tmp,2);
	// 			transformInd = tmp[Fx][Fy] + 1;
	// 		else
	// 			transformInd = floor(numPartRotate/2) + 1;
			
			
	// 		actualPartRotationInd = transformInd - numPartRotate*(ceil(double(transformInd)/numPartRotate)-1);
	// 		Fx = floor( Fx + M2RowColShift[translationInd][0] * partSizeX/subsampleS2 );
	// 		Fy = floor( Fy + M2RowColShift[translationInd][1] * partSizeY/subsampleS2 );
	// 	else
	// 		actualPartRotationInd = r;
		
	// 	actualPartRotation = partRotationRange(actualPartRotationInd);

	// 	% find the part location at the higher resolution
	// 	Fx = (Fx-1 + .5) * subsampleS2 * subsampleM2;
	// 	Fy = (Fy-1 + .5) * subsampleS2 * subsampleM2;
	// 	partAbsoluteLocation[iPart][0] = Fx;
	// 	partAbsoluteLocation[iPart][1] = Fy;
	// 	partAbsoluteResolution[iPart] = bestPartRes;
	// 	partAbsoluteRotation[iPart] = actualPartRotation;

	// 	if(doMorphBackS1map)
	        
	//         % some precomputation
	//         denseX = -floor(partSizeX/2) + (1:partSizeX);
	//         denseY = -floor(partSizeY/2) + (1:partSizeY);
	//         count = 0;
	//         inRow = zeros(length(denseX)*length(denseY),1,'single');
	//         inCol = zeros(length(denseX)*length(denseY),1,'single');
	//         for y = denseY
	//             for x = denseX
	//                 count = count+1;
	//                 inRow(count) = x;
	//                 inCol(count) = y;
	//             end
	//         end
	//         tScale = 0; rScale = 1; cScale = 1; inO = zeros(numel(inRow),1,'single'); inS = zeros(numel(inRow),1,'single');
	//         [outRow, outCol] = ...
	//             mexc_TemplateAffineTransform(tScale,rScale,cScale,...
	//                 actualPartRotation,inRow,inCol,inO,inS,numOrient);
	        
	// 		% crop the feature patch that is registered to the part template
	        
	// 		tmpCropped = mexc_CropInstance(SUM1mapFind(bestPartRes,:),Fx,Fy,...
	// 			actualPartRotation,tScale,1,...
	// 			outRow,outCol,...
	// 			numOrient,1,partSizeX,partSizeY);
	// 		for o = 1:numOrient
	// 			morphedSUM1map{o}(PartLocX(iPart)-1+(1:partSizeX),PartLocY(iPart)-1+(1:partSizeY)) = tmpCropped{o};
	// 		end
	// 		% also crop the corresponding image patch (for each part)
	// 		tmpCropped = mexc_CropInstance(ImageMultiResolution(bestPartRes),Fx,Fy,...
	// 			actualPartRotation,tScale,1,...
	// 			outRow,outCol,...
	// 			1,1,partSizeX,partSizeY);
	// 		morphedPatch(PartLocX(iPart)-1+(1:partSizeX),PartLocY(iPart)-1+(1:partSizeY)) = tmpCropped{1};
	// 	end
		
	// 	% ==== continue to trace back Gabor elements based on the part localization ====

	// 	gaborXX = [];
	// 	gaborYY = [];
	// 	gaborOO = [];
	// 	gaborMM = [];

	// 	% Gabor basis elements locations
	// 	for j = 1:length( selectedlambda{iPart} )
	// 		gaborX = floor(Fx + allSelectedx{iPart,actualPartRotationInd}(j));
	// 		gaborY = floor(Fy + allSelectedy{iPart,actualPartRotationInd}(j));
	// 		gaborO = allSelectedOrient{iPart,actualPartRotationInd}(j);
	// 		if gaborX > 0 && gaborX <= size(M1Trace{bestPartRes,1},1) && gaborY > 0 && gaborY <= size(M1Trace{bestPartRes,1},2)
	// 			trace = M1Trace{bestPartRes,gaborO+1}(gaborX,gaborY) + 1;
	// 			dx = M1RowShift{gaborO+1}(trace);
	// 			dy = M1ColShift{gaborO+1}(trace);
	// 			shiftedo = M1OriShifted{gaborO+1}(trace);
	// 			gaborX = floor(.5 + gaborX + single(dx));
	// 			gaborY = floor(.5 + gaborY + single(dy));
	// 			gaborO = single(shiftedo);
	// 		end
	// 		gaborXX = [gaborXX;gaborX];
	// 		gaborYY = [gaborYY;gaborY];
	// 		gaborOO = [gaborOO;gaborO];
	// 		gaborCount = gaborCount + 1;
	// 		gaborAbsoluteLocation(gaborCount,1) = gaborX;
	// 		gaborAbsoluteLocation(gaborCount,2) = gaborY;
	// 		gaborAbsoluteRotation(gaborCount) = gaborO; % start from 0
	// 		gaborAbsoluteResolution(gaborCount) = bestPartRes;
	// 		if gaborX > 0 && gaborX <= size(M1Trace{bestPartRes,1},1) && gaborY > 0 && gaborY <= size(M1Trace{bestPartRes,1},2)
	// 			val = SUM1mapFind{bestPartRes,gaborO+1}(gaborX,gaborY);
	// 		else
	// 			val = 0;
	// 		end
	// 		gaborMM = [gaborMM; max(0,sqrt(val)-.2)];
	// 		gaborResponses(gaborCount) = val;
	// 	end
		
	// 	if showMatchedTemplate
	// 		% render the template for each part separately, then overlay the rendered images
	// 		tmpMatchedSym = displayMatchedTemplate(size(ImageMultiResolution{bestPartRes}),gaborXX,...
	// 			gaborYY,gaborOO,zeros(length(gaborXX),1,'single'),gaborMM,allSymbol,numOrient);
	// 		tmpMatchedSym = double( imresize(tmpMatchedSym,imageSizeAtBestObjectResolution,'bilinear') );
	// 		matchedSym = max(matchedSym,tmpMatchedSym);
	// 		if showPartBoundingBox
	//             margin = 3;
	//             xx = repmat((1:partSizeX),1,margin*2);
	//             yy = [];
	//             for y = [1:margin partSizeY-margin+1:partSizeY]
	//                 yy = [yy,ones(1,partSizeX)*y];
	//             end
	//             yy = [yy,repmat((1:partSizeY),1,margin*2)];
	//             for x = [1:margin partSizeX-margin+1:partSizeX]
	//                 xx = [xx,ones(1,partSizeY)*x];
	//             end
	//             inRow = single(xx-floor(partSizeX/2)); inCol = single(yy-floor(partSizeY/2));
	//             tScale = 0; rScale = 1; cScale = 1; inO = zeros(numel(inRow),1,'single'); inS = zeros(numel(inRow),1,'single');
	//             [outRow, outCol] = ...
	//                 mexc_TemplateAffineTransform(tScale,rScale,cScale,...
	//                     actualPartRotation,inRow,inCol,inO,inS,numOrient);
	                
	//             % directly overwrite the corresponding pixels
	//             matchedBoundingBox = imresize(matchedBoundingBox,size(ImageMultiResolution{bestPartRes}),'nearest');
	//             for p = 1:length(outRow)
	//                 x = floor(.5 + outRow(p) + Fx); y = floor(.5 + outCol(p) + Fy);
	//                 if x > 0 && x <= size(matchedBoundingBox,1) && y > 0 && y <= size(matchedBoundingBox,2)
	//                     matchedBoundingBox(x,y,:) = [1 .5 .3];
	//                 end
	//             end
	//             matchedBoundingBox = imresize(matchedBoundingBox,size(ImageMultiResolution{bestRes}),'nearest');
	// 		end
	// 	end
	// }

//}
}
}

