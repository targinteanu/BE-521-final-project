

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
Yactive = Ymm > .5; 
Y0 = Y; Y = double(Yactive) + 1;

% downsample
X = MU;
%ds = 100;
%X = X(1:ds:end,:); Y = Y(1:ds:end,:);
ds = floor(length(Y)/length(X));
Y = Y(1:ds:end,:);
trim = length(Y)-length(X) + 1;
Y = Y(trim:end,:);
Y0 = Y0(1:ds:end,:); Y0 = Y0(trim:end,:);

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

%%
%{
figure; clear ax;
Xc = zeros(size(Y));
w = zeros(610,5);
spars = zeros(1,5); stren = spars; eps = spars;
trainbound = floor(.8*length(X));
for f = 1:5
    ax(f) = subplot(5,1,f);
    plot(Y(:,f), 'k', 'LineWidth', 1); hold on; grid on; 
    plot(Y0(:,f), 'k');
    [~,w(:,f),spars(f),stren(f),eps(f),mag,ang] = ...
        YOLC(X(1:trainbound,:), Y(1:trainbound,f), -.1, 0, 1e-3, 0, true);
    Xc(:,f) = X*w(:,f);
    plot(Xc(:,f), 'b');
    f
end
linkaxes(ax); clear ax;

%%
m1 = arrayfun(@(f) mean(Xc(Y(1:trainbound,f)==1,f)), 1:5); s1 = arrayfun(@(f) std(Xc(Y(1:trainbound,f)==1,f)), 1:5);
m2 = arrayfun(@(f) mean(Xc(Y(1:trainbound,f)==2,f)), 1:5); s2 = arrayfun(@(f) std(Xc(Y(1:trainbound,f)==2,f)), 1:5);
figure; errorbar(m1, s1); hold on; errorbar(m2, s2); grid on;
thresh = (m1 + m2)/2; [~,dir] = max([m1;m2]);
figure; clear ax;
for f = 1:5
    ax(f) = subplot(5,1,f);
    plot(Y(:,f), 'k', 'LineWidth', 1); hold on; grid on; 
    plot(Y0(:,f), 'k');
    if dir(f) == 1
        Xbin = (Xc(:,f) < thresh(f));
    else
        Xbin = (Xc(:,f) > thresh(f));
    end
    Xbin = Xbin + 1;
    plot(Xbin, ':b', 'LineWidth', 2);
    acc = mean(Xbin == Y(:,f));
    trainacc = mean(Xbin(1:trainbound) == Y(1:trainbound,f));
    testacc = mean(Xbin(trainbound:end) == Y(trainbound:end,f));
    title([num2str(acc) ' | train ' num2str(trainacc) ' | test ' num2str(testacc)]);
end
linkaxes(ax); clear ax;

%}

%%
figure; clear ax;
Xc = zeros(size(Y));
w = zeros(610,5);
spars = zeros(1,5); stren = spars; eps = spars;
trainbound = floor(.8*length(X));
for f = 1:5
    Xtr = X(1:trainbound,:); Ytr = Y(1:trainbound,f); Y0tr = Y0(1:trainbound,f);
    Xtr = Xtr(Ytr == 2); Y0tr = Y0tr(Ytr == 2);
    ax(f) = subplot(5,1,f);
    plot(Y(:,f), 'k', 'LineWidth', 1); hold on; grid on; 
    plot(Y0(:,f), 'k');
    [~,w(:,f),spars(f),stren(f),eps(f),mag,ang] = ...
        YOLC(Xtr, Y0tr, -.1, 0, 1e-3, 0, false);
    Xc(:,f) = X*w(:,f);
%    plot(Xc(:,f), 'b');
    f
end
linkaxes(ax); clear ax;
figure; grid on; hold on;
for f = 1:5
    plot(Xc(Y(:,f)==2,f), Y0(Y(:,f)==2,f), '.');
end