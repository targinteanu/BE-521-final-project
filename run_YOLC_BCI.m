% load raw EEG data 
sub = 1; 
load('project_data_edit.mat')
Xraw = train_ecog{sub}; Yraw = train_dg{sub};
load('mufilter.mat') % Hbp
load('bandfilters.mat') % band1filt ... band5filt
fs = 1000; % Hz

% define feature functions 
LL = @(x) sum(abs(diff(x)));
Area = @(x) sum(abs(x));
Energy = @(x) sum(x.^2);
ZX = @(x) sum( ((x(1:end-1)-mean(x)>0)&(x(2:end)-mean(x)<0)) | ...
    ((x(1:end-1)-mean(x)<0)&(x(2:end)-mean(x)>0)) );
NumWins = @(xLen, fs, winLen, winDisp) floor((xLen/fs - winLen + winDisp)/winDisp);

% extract mu brainwave and other bands 
bandfilts = [Hbp, band1filt, band2filt, band3filt, band4filt, band5filt];
Xband = cell(size(bandfilts));
for filt = 1:length(bandfilts)
    Xband{filt} = filter(bandfilts(filt), Xraw);
end
Xband = [Xraw, Xband]; % include unfiltered 

% define window size and displacement
winsize = 100; % ms
windisp = 50; % ms
numwins = 10; % # of prev windows to use at each time (?)

% get power (~time RMS integral) of each band in each window
XbandWin = cell(size(Xband));
for band = 1:length(Xband)
    M = Xband{filt};
    M = sqrt(movmean(M.^2, winsize)); % RMS integral 
    M2 = M(1:windisp:end,:); % window 
    M3 = zeros( size(M2).*[1,numwins] - [numwins,0] );
    for t = 1:length(M3)
        wins = M2(t:(t+numwins-1),:);
        M3(t,:) = wins(:)';
    end
    XbandWin{band} = M3;
end

%% get other windowed features
featfn = {LL, Area, Energy, ZX};
Xfeats = arrayfun(@(i) MovingWinFeats(Xraw, fs, winsize/1000, windisp/1000, featfn{i}), ...
    1:length(featfn), 'UniformOutput', false);
