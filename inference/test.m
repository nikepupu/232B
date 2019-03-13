% The detection module for Hierarchical Active Basis.
%
% It takes the HAB model files and a JPEG source image or filtered image as input.
%
% It outputs the following information:
% 1) The hierarchical deformation of HAB: a tree with placements (location, orierntation, scale) of the object, parts and edge elements.
% 2) The hierarchical score of HAB: a tree with scores or responses of the object, parts and edge elements.
% 3) The visualization of deformed HAB on the observed image. 
% 4) (optional) The cropped Gabor filter response map and image patch by morphing back from the observed response map or image.

% paths
modelFolder = 'working';
configFile = 'Config.mat';
dotFile = 'inferenceResult.dot';
imageFolder = 'positiveImage';
imageName = 'l29.jpg';
featureFolder = 'working';
featurefile = sprintf('%s/%s.mat',featureFolder,imageName);
reLoadModel = false; % if true -> enforce the model to be reloaded
it = 25;

% parameters for "morphing-back"
doMorphBackS1map = true;
doCropBackImage = true;

% parameters for visualization of deformed HAB
showMatchedTemplate = true;
showPartBoundingBox = true;
showObjectBoundingBox = true;

%% load model and configuration file if necessary
if ~exist('allS3Selectedx','var') || reLoadModel
	load( configFile, 'subsampleS2', 'numOrient', 'allFilter', 'allSymbol', 'partRotationRange', 'locationPerturbFraction', 'numPartRotate',...
		'maxPartRelativeRotation', 'resolutionShiftLimit', 'numCandPart', 'PartLocX', 'PartLocY', 'sizeTemplatex', 'sizeTemplatey',...
		'partSizeX', 'partSizeY', 'minRotationDif', 'rotationRange', 'numResolution', 'resizeFactor', 'numElement',...
        'localHalfx', 'localHalfy', 'thresholdFactor', 'saturation', 'locationShiftLimit', 'orientShiftLimit' );
	
	load( sprintf('%s/partModel_iter%d.mat',modelFolder,it),...
		'allSelectedx', 'allSelectedy', 'allSelectedOrient',...
        'selectedlambda', 'selectedLogZ','largerAllSelectedx','largerAllSelectedy','largerAllSelectedOrient','largerSelectedlambda','largerSelectedLogZ');
    
	load(sprintf('%s/objectModel_iter%d.mat',modelFolder,it),...
		'PartOnOff', 'allS3SelectedRow', 'allS3SelectedCol', 'allS3SelectedOri');
end

FILE2= ["allSelectedx", "allSelectedy", "allSelectedOrient",...
        "selectedlambda", "selectedLogZ","largerAllSelectedx","largerAllSelectedy","largerAllSelectedOrient","largerSelectedlambda","largerSelectedLogZ"];
FILE3 = [
    "PartOnOff", "allS3SelectedRow", "allS3SelectedCol", "allS3SelectedOri"
];
selectedPart = find(PartOnOff);
% this seems to be column major
sz = size(allSelectedx);
ssz = sz(1)*sz(2);
data1 = jsonencode(allSelectedx);

for name = FILE2
    
temp = eval(name);
sz = size(temp);
output = strcat(num2str(sz(1)) , " " , num2str(sz(2)) , '\n');
for i = 1:sz(1)
    for j = 1:sz(2)
       
        data = temp{i, j};
        sz2 = size(data);
        output = strcat(output, num2str(sz2(1)) , " " , num2str(sz2(2)) , '\n' );
        if sz2(1) == 0 || sz2(2) == 0
            output = strcat(output, '\n');
            continue
        end
        if sz2(1) == 1 || sz2(2) == 1
            for m = 1:max(sz2(1),sz2(2))
             output = strcat(output, num2str(data(m))," ");
            end
        else
        
            for m = 1:sz2(1)
             for n = 1:sz2(2)
                    output = strcat(output, num2str(data(m,n)), " ");
             end
            end
        end
        
        output = strcat(output, '\n');
    end
end
fileID = fopen(strcat(name,'.txt'),'w');
fprintf(fileID, output);
end


for name = FILE3
temp = eval(name);
sz = size(temp);
output = strcat(num2str(sz(1)) , " " , num2str(sz(2)) , '\n');
for i = 1:sz(1)
    for j = 1:sz(2)
        data = temp(i,j);
        output = strcat(output, num2str(data) , " " );
    end
end
output = strcat(output, '\n');
fileID = fopen(strcat(name,'.txt'),'w');
fprintf(fileID, output);
end

