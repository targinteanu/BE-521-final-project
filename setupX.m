

sub = 1; 
load('project_data_edit.mat')
X = train_ecog{sub}; Y = train_dg{sub};
load('mufilter.mat')

%%
mu_raw = filter(Hbp, X);

%%
MU = mu_raw;
winsize = 1000; % ms
windisp = 100; % ms
numwins = 10;
MU = sqrt(movmean(MU.^2, winsize));

MU2 = MU(1:windisp:end,:);
MU3 = zeros( size(MU2).*[1,numwins] - [numwins,0] );
for t = 1:length(MU3)
    wins = MU2(t:(t+numwins-1),:);
    MU3(t,:) = wins(:)';
end
MU = MU3;


Y = train_dg{sub};
Ymm = movmean(Y, ceil(length(Y)/100));
Yactive = Ymm > .1; 
Y0 = Y; Y = double(Yactive) + 1;

% downsample
X = MU;
%ds = 100;
%X = X(1:ds:end,:); Y = Y(1:ds:end,:);
ds = floor(length(Y)/length(X));
Y = Y(1:ds:end,:);
trim = length(Y)-length(X) + 1;
Y = Y(trim:end,:);

% normalize X
%X = X - mean(X, 2); X = X./std(X, [], 2);
%Y = Y - mean(Y, 2); Y = Y./std(Y, [], 2);
X = X - mean(X);

%[C,S,~,~,pe] = pca(X); figure; plot(pe)

onIdx = false(size(Y)); offIdx = false(size(Y));
Xon = cell(1,5); Xoff = cell(1,5);
C = cell(1,5);
for fing = 1:5
    onIdx(:,fing) = Y(:,fing) == 2; offIdx(:,fing) = Y(:,fing) == 1;
    Xon{fing} = X(onIdx(:,fing),:); Xoff{fing} = X(offIdx(:,fing),:);
    C{fing} = pca([Xon{fing}; Xoff{fing}]);
end