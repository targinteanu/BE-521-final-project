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

% get power (~time RMS integral) of each band in each window
XbandWin = cell(size(Xband));
for band = 1:length(Xband)
    M = Xband{filt};
    M = sqrt(movmean(M.^2, winsize)); % RMS integral 
    M2 = M(1:windisp:end,:); % window 
%    M3 = zeros( size(M2).*[1,numwins] - [numwins,0] );
%    for t = 1:length(M3)
%        wins = M2(t:(t+numwins-1),:);
%        M3(t,:) = wins(:)';
%    end
    XbandWin{band} = M2;
end

% get other windowed features
featfn = {LL, Area, Energy, ZX};
Xfeats = arrayfun(@(i) MovingWinFeats(Xraw, fs, winsize/1000, windisp/1000, featfn{i}), ...
    1:length(featfn), 'UniformOutput', false);

%% combine into single windowed matrix 
Xwin = [XbandWin, Xfeats]; % cell array 
durs = arrayfun(@(i) length(Xwin{i}), 1:length(Xwin)); % make durations equal
dur = min(durs);
Xwin = arrayfun(@(i) Xwin{i}(1:dur,:), 1:length(Xwin), 'UniformOutput', false);
XwinM = cell2mat(Xwin);

% use previous <numwins> windows as features at each time 
numwins = 3; % # of prev windows to use at each time
X = zeros( size(XwinM).*[1,numwins] - [numwins,0] );
for t = 1:size(X,1)
    wins = XwinM(t:(t+numwins-1),:);
    X(t,:) = wins(:)';
end

%% set up Y
Ymm = movmean(Yraw, ceil(length(Yraw)/100));
Ybin = Ymm > .5; % active vs inactive 
Ybin = double(Ybin) + 1;

Y = movmean(Yraw, winsize); % downsampled 
Y = Y(1:windisp:end,:); % windowed 
Y = Y(1:dur,:); % trimmed 
Y = Y((numwins+1):end,:); % delay 

%%
fing = 3; 
y = Y(:,fing);
idxOn = y == 2; idxOff = y == 1;

[Xc, w, sparsity, strength, eps, mags, angles] = YOLC(X, y, -.1, 0, 1e-3, 0, false);
figure; plot(Xc, y, '.');