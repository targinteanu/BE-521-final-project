% load raw EEG data 
sub = 1; 
load('project_data_edit.mat')
Xraw = train_ecog{sub}; Yraw = train_dg{sub};
%load('mufilter.mat') % Hbp
%load('bandfilters.mat') % band1filt ... band5filt
fs = 1000; % Hz

%%
NumWins = @(xLen, fs, winLen, winDisp) floor((xLen/fs - winLen + winDisp)/winDisp);
%{

% define feature functions 
LL = @(x) sum(abs(diff(x)));
Area = @(x) sum(abs(x));
Energy = @(x) sum(x.^2);
ZX = @(x) sum( ((x(1:end-1)-mean(x)>0)&(x(2:end)-mean(x)<0)) | ...
    ((x(1:end-1)-mean(x)<0)&(x(2:end)-mean(x)>0)) );

% extract mu brainwave and other bands 
%%{
%bandfilts = [Hbp, band1filt, band2filt, band3filt]; %, band4filt, band5filt];
%bandfilts = [Hbp, band1filt, band3filt];
bandfilts = [Hbp];
Xband = cell(size(bandfilts));
for filt = 1:length(bandfilts)
    Xband{filt} = filter(bandfilts(filt), Xraw);
end
%Xband = [Xraw, Xband]; % include unfiltered 
%%}
%Xband = {Xraw};

%}

% define window size and displacement
winsize = 100; % ms
windisp = 50; % ms

%{
% get power (~time RMS integral) of each band in each window
XbandWin = cell(size(Xband));
for filt = 1:length(Xband)
    M = Xband{filt};
    M = sqrt(movmean(M.^2, winsize)); % RMS integral 
    M2 = M(1:windisp:end,:); % window 
%    M3 = zeros( size(M2).*[1,numwins] - [numwins,0] );
%    for t = 1:length(M3)
%        wins = M2(t:(t+numwins-1),:);
%        M3(t,:) = wins(:)';
%    end
    XbandWin{filt} = M2;
end

% avg signal
Mraw = movmean(Xraw, winsize);
M2 = Mraw(1:windisp:end,:);
%Xband = [Mraw, Xband];

% get other windowed features
%featfn = {LL, Area, Energy, ZX};
featfn = {LL};
Xfeats = arrayfun(@(i) MovingWinFeats(Xraw, fs, winsize/1000, windisp/1000, featfn{i}), ...
    1:length(featfn), 'UniformOutput', false);

%% combine into single windowed matrix 
Xwin = [XbandWin, Xfeats]; % cell array 
durs = arrayfun(@(i) length(Xwin{i}), 1:length(Xwin)); % make durations equal
dur = min(durs);
Xwin = arrayfun(@(i) Xwin{i}(1:dur,:), 1:length(Xwin), 'UniformOutput', false);
Xwin = arrayfun(@(i) (Xwin{i} - mean(Xwin{i}(:)))/std(Xwin{i}(:)), ...
    1:length(Xwin), 'UniformOutput', false); % normalize 
XwinM = cell2mat(Xwin);

%}

%{
% downsample 
ds = 40; 
%Yds = Yraw((ds/2):ds:end,:); 
%Yds = Yraw(ds:ds:end,:); 
Ymm1 = movmean(Yraw, ds);
ds = 10; 
Yds = Ymm1(ds:ds:end,:);

dsX = floor(ds/numwins);
%Xmm = movmean(Xraw, ds);
%Xds = Xmm((ds/2):ds:end,:);
Xds = Xraw(1:dsX:end,:);
%}

X1 = zeros( NumWins(size(Xraw,1), 1, winsize, windisp), size(Xraw,2) );
for i = 1:size(Xraw,2)
    [s,f,t,p] = spectrogram(Xraw(:,i), winsize, windisp, 2*fs, fs);
    [q,nd] = max(p);
    X1(:,i) = f(nd);
end

% use previous <numwins> windows as features at each time 
numwins = 5; % # of prev windows to use at each time
X = zeros( size(X1).*[1,numwins] - [numwins,0] );
for t = 1:size(X,1)
    wins = X1(t:(t+numwins-1),:);
    X(t,:) = wins(:)';
end

% downsample X
%X = X(1:numwins:end,:);

% investigate X
rho = corr(X); figure; imshow(rho);
[C,S,~,~,pe] = pca(X); %figure; plot(pe);

%{
% set up Y
Ymm = movmean(Y, ceil(length(Y)/100));
Ybin = Ymm > .5; % active vs inactive 
Ybin = double(Ybin) + 1;
%Y = Yds;
%}
dur = size(X,1);
Ymm = movmean(Yraw, ceil(length(Yraw)/100));
Ybin = Ymm > .5; % active vs inactive 
Ybin = double(Ybin) + 1;

Y = movmean(Yraw, winsize); % downsampled 
Y = Y(1:windisp:end,:); Ybin = Ybin(1:windisp:end,:); % windowed 
Y = Y(1:dur,:); Ybin = Ybin(1:dur,:); % trimmed 
Y = Y((numwins+1):end,:); Ybin = Ybin((numwins+1):end,:); % delay 
%Y = Y(2:end,:); Ybin = Ybin(2:end,:); % delay 

%%
%%{
fing = 3; 
y = Y(:,fing);
ybin = Ybin(:,fing);
idxOn = ybin == 2; idxOff = ybin == 1;

[Xc, w, sparsity, strength, eps, mags, angles, Xdist] = YOLC(X(idxOn,:), y(idxOn), 200, 0, .1, 0, false);

%
%figure; plot(y/max(y(:))); hold on; plot(ybin-1); plot(X*w/max(X(:)));
figure; plot(Xc, y(idxOn), '*'); hold on; grid on;
%[sparsity, strength, eps]
%figure; histogram(Xdist(:));
%}

%% random cross val 
IdxOn = find(idxOn);
trainsz = ceil(.5*length(IdxOn));
%IdxTrain = randperm(length(IdxOn)); IdxTrain = IdxTrain(1:trainsz); 
IdxTrain = 1:trainsz; 
IdxTrain2 = IdxOn(IdxTrain);
[xci, wi, spi, stri, epsi, gwi, angi, Xdi] = YOLC(X(IdxTrain2,:), y(IdxTrain2), 200, 0, .1, 0, false);
xc = X(idxOn,:)*wi;
FO = fit(xci, y(IdxTrain2), 'poly1'); 
ypred = xc*FO.p1 + FO.p2;
ytrainpred = nan(size(ypred)); ytrainpred(IdxTrain) = xc(IdxTrain)*FO.p1 + FO.p2;
idxTest = isnan(ytrainpred); IdxTest2 = IdxOn(idxTest);
figure; plot(y(idxOn), 'k', 'LineWidth', 2); grid on; hold on; 
plot(ytrainpred, 'r', 'LineWidth', 1);
plot(ypred, '--r');
title(['\rho_{tot} = ' num2str(corr(y(idxOn), ypred)) ...
    ', \rho_{tr} = ' num2str(corr(y(IdxTrain2), ytrainpred(IdxTrain))) ...
    ', \rho_{te} = ' num2str(corr(y(IdxTest2), ypred(idxTest))) ]);
legend('actual', 'training', 'testing')

%% cross validate 
%{
IdxOn = find(idxOn);
trainsz = ceil(.8*length(IdxOn));
for i = 1:25:(length(IdxOn) - trainsz)
    IdxTrain = IdxOn( i + (0:(trainsz - 1)) ); 
    [xci, wi, spi, stri, ~,~,~, Xdi] = YOLC(X(IdxTrain,:), y(IdxTrain), eps, 0, 1e-3, 0, false);
    plot(X(IdxOn,:)*wi, y(IdxOn), '.');
    [i, spi, stri]
end
%}