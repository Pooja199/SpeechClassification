addpath(genpath('utility'));
addpath(genpath('RemoveWhiteSpace'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));

% set paths to the wave files and protocols
pathToDatabase = fullfile('C:\Users\Shrirag\Desktop','ASV Trimmed Data');

evalProtocolFile = fullfile('C:\Users\Shrirag\Desktop','ASV Trimmed Data','ASVspoof2017_eval.trl');

fileID = fopen(evalProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
disp('Computing scores parparfor evaluation trials...');
evalFeatureCell1 = eye(length(filelist),7020);
parfor i=(1:length(filelist))
    filePath = fullfile(pathToDatabase,'eval_trimmed',filelist{i});
    [x,fs] = audioread(filePath);
    a = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZD');
    a = a(:);
    newa = padarray(a,7020-length(a),0,'post');
    %newa = reshape(newa,30,887);
    evalFeatureCell1(i,:) = newa';
end

evalFeatureCell2 = [evalFeatureCell ; evalFeatureCell1];

disp('Fitting');
yfit = TreeWithDev.predictFcn(evalFeatureCell2);