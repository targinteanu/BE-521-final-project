%%
MU = mu_raw;
winsize = 500; % ms
windisp = 100; % ms
numwins = 5;
MU = sqrt(movmean(MU.^2, winsize));

MU2 = MU(1:windisp:end,:);
MU3 = zeros( size(MU2).*[1,numwins] - [numwins,0] );
for t = 1:length(MU3)
    wins = MU2(t:(t+numwins-1),:);
    MU3(t,:) = wins(:)';
end
MU = MU3;

%%
fing = 2;
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
%X = X - mean(X);

%[C,S,~,~,pe] = pca(X); figure; plot(pe)

onIdx = Y(:,fing) == 2; offIdx = Y(:,fing) == 1;
%Son = S(onIdx,:); Soff = S(offIdx,:);
Xon = X(onIdx,:); Xoff = X(offIdx,:);

%%
XonT = tsne(Xon); XoffT = tsne(Xoff);
figure; plot(XonT(:,1), XonT(:,2), 'o');
hold on; grid on; plot(XoffT(:,1), XoffT(:,2), 'x');