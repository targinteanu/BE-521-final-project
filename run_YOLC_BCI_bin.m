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
%%{
%bandfilts = [Hbp, band1filt, band2filt, band3filt]; %, band4filt, band5filt];
%bandfilts = [Hbp, band1filt, band3filt];
bandfilts = [Hbp];
Xband = cell(size(bandfilts));
for filt = 1:length(bandfilts)
    Xband{filt} = filter(bandfilts(filt), Xraw);
end
%Xband = [Xraw, Xband]; % include unfiltered 
%}
%Xband = {Xraw};

% define window size and displacement
winsize = 100; % ms
windisp = 50; % ms

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

% use previous <numwins> windows as features at each time 
numwins = 3; % # of prev windows to use at each time
X = zeros( size(XwinM).*[1,numwins] - [numwins,0] );
for t = 1:size(X,1)
    wins = XwinM(t:(t+numwins-1),:);
    X(t,:) = wins(:)';
end

%% investigate X
rho = corr(X); figure; imshow(rho);
[C,S,~,~,pe] = pca(X); %figure; plot(pe);

%% set up Y
Ymm = movmean(Yraw, ceil(length(Yraw)/100));
Ybin = Ymm > .5; % active vs inactive 
Ybin = double(Ybin) + 1;

Y = movmean(Yraw, winsize); % downsampled 
Y = Y(1:windisp:end,:); Ybin = Ybin(1:windisp:end,:); % windowed 
Y = Y(1:dur,:); Ybin = Ybin(1:dur,:); % trimmed 
Y = Y((numwins+1):end,:); Ybin = Ybin((numwins+1):end,:); % delay 

%%
%%{
fing = 3; 
y = Y(:,fing);
ybin = Ybin(:,fing);
idxOn = ybin == 2; idxOff = ybin == 1;

%[Xc, w, sparsity, strength, eps, mags, angles, Xdist] = YOLC(X, ybin, 30, 0, 1e-3, 0, true);

%
%figure; plot(y/max(y(:))); hold on; plot(ybin-1); plot(X*w/max(X(:)));
%figure; plot(Xc, ybin, '*'); hold on; grid on;
%[sparsity, strength, eps]
%figure; histogram(Xdist(:));
%}

%% random cross val 
IdxOn = find(idxOn);
trainsz = ceil(.5*length(y));
%IdxTrain = randperm(length(IdxOn)); IdxTrain = IdxTrain(1:trainsz); 
IdxTrain = 1:trainsz; 
[xci, wi, spi, stri, epsi ,gwi,angi, Xdi] = YOLC(X(IdxTrain,:), ybin(IdxTrain), 30, 0, 1e-3, 0, true);

mns = [mean(xci(ybin(IdxTrain)==1)), mean(xci(ybin(IdxTrain)==2))]; 
[~,i] = max(mns);

xc = X*wi;
%FO = fit(xci, y(IdxTrain), 'poly1'); 
if i == 2
    ypred = xc > mean(mns);
else
    ypred = xc <= mean(mns);
end
ypred = ypred + 1;

ytrainpred = nan(size(ypred)); ytrainpred(IdxTrain) = ypred(IdxTrain);
idxTest = isnan(ytrainpred); 
figure; plot(ybin, 'k', 'LineWidth', 2); grid on; hold on; 
plot(ytrainpred, 'r', 'LineWidth', 1);
plot(ypred, '--r');

title(['acc_{tot} = ' num2str(mean( ypred == ybin )) ...
    ', acc_{tr} = ' num2str(mean( ypred(IdxTrain) == ybin(IdxTrain) )) ...
    ', acc_{te} = ' num2str(mean( ypred(idxTest) == ybin(idxTest) )) ]);

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