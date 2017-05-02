%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASVspoof 2017 CHALLENGE:
% Audio replay detection challenge parfor automatic speaker verification anti-spoofing
% 
% http://www.spoofingchallenge.org/
% 
% ====================================================================================
% Matlab implementation of the baseline system parfor replay detection based
% on constant Q cepstral coefficients (CQCC) features + Gaussian Mixture Models (GMMs)
% ====================================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;

% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));

% set paths to the wave files and protocols
pathToDatabase = fullfile('C:\Users\Pooja\Desktop\baseline_CM','ASVspoof2017_train_dev');
trainProtocolFile = fullfile('C:\Users\Pooja\Desktop\baseline_CM\ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_train.trn');
devProtocolFile = fullfile('C:\Users\Pooja\Desktop\baseline_CM\ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_dev.trl');


% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Feature extraction parfor training data

% extract features parfor GENUINE training data and store in cell array
disp('Extracting features parfor GENUINE training data...');
genuineFeatureCell = eye(length(genuineIdx),7020);
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'wav\train_trimmed',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    a = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZD');
    a = a(:);
    newa = padarray(a,7020-length(a),0,'post');
    %newa = reshape(newa,30,887);
    genuineFeatureCell(i,:) = newa';
end
disp('Done!');

% extract features parparfor SPOOF training data and store in cell array
disp('Extracting features parparfor SPOOF training data...');
spoofFeatureCell = eye(length(spoofIdx),7020);
for i=1:length(spoofIdx)
    if(i~=2970)
        filePath = fullfile(pathToDatabase,'wav\train_trimmed',filelist{spoofIdx(i)});
        [x,fs] = audioread(filePath);
        a = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZD');
        a = a(:);
        newa = padarray(a,7020-length(a),0,'post');
        %newa = reshape(newa,30,887);
        spoofFeatureCell(i,:)= newa';
    end
end
disp('Done!');

%GFC_3D = cat(3,genuineFeatureCell{:});
%GFC = reshape(GFC_3D, size(GFC_3D,1) * size(GFC_3D,2),[]);

%SFC_3D = cat(3,spoofFeatureCell{:});
%SFC = reshape(SFC_3D, size(SFC_3D,1) * size(SFC_3D,2),[]);

X = [genuineFeatureCell ; spoofFeatureCell];

%Y = [genuineIdx ; spoofIdx];

%SVMModel = fitcsvm(X,labels,'KernelFunction','rbf','Standardize',true,'ClassNames',{'spoof','genuine'});

binarylabels = zeros(3015,1);
for i=1:length(labels)
    if(strcmp(labels{i},'genuine'))
        binarylabels(i) = 1;
    end
end
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};
disp('Computing scores parparfor development trials...');

x_cqcc = eye(length(filelist),7020);
parfor i=(1:length(filelist))
    filePath = fullfile(pathToDatabase,'wav\dev_trimmed',filelist{i});
    [x,fs] = audioread(filePath);
    info = audioinfo(filePath);
   
    % featrue extraction
        a = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZD');
        a = a(:);

        newa = padarray(a,7020-length(a),0,'post');
        %newa = reshape(newa,30,887);
        x_cqcc(i,:) = newa';
   
end

binarylabels2 = zeros(1710,1);
for i=1:length(labels)
    if(strcmp(labels{i},'genuine'))
        binarylabels2(i) = 1;
    end
end

%blabels = [binarylabels ; binarylabels2];
%X = [X ; x_cqcc];
%Xvar = [X blabels];

%x_3D = cat(3,x_cqcc{:});
%x = reshape(x_3D, size(x_3D,1) * size(x_3D,2),[]);

%[label,score] = predict(SVMModel,x_cqcc);

%save('CQCCvariablesNaN.mat','x_cqcc','X');




