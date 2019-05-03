

sub = 1; 
load('project_data_edit.mat')
X = train_ecog{sub}; Y = train_dg{sub};
load('mufilter.mat')

%%
%%{
neur = 49; 
sig = X(:,neur);
sigmu = filter(Hbp, sig);
feat = sqrt(movmean(sigmu.^2, 1000));
%}

%%

figure; clear ax; 
for f = 1:5
    ax(f) = subplot(6,1,f);
    plot(Y(:,f)); 
    grid on;
end
ax(6) = subplot(6,1,6); 
plot(feat); grid on;
linkaxes(ax, 'x'); 
clear ax;

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
X = X - mean(X);

%[C,S,~,~,pe] = pca(X); figure; plot(pe)

onIdx = Y(:,fing) == 2; offIdx = Y(:,fing) == 1;
%Son = S(onIdx,:); Soff = S(offIdx,:);
Xon = X(onIdx,:); Xoff = X(offIdx,:);

%{
cols = [121, 261, 331];
figure; plot3(Xon(:,cols(1)), Xon(:,cols(2)), Xon(:,cols(3)), 'o');
hold on; grid on; plot3(Xoff(:,cols(1)), Xoff(:,cols(2)), Xoff(:,cols(3)), 'x');
xlabel(num2str(cols(1))); ylabel(num2str(cols(2))); zlabel(num2str(cols(3))); 
%}

%%
[C,S,~,~,pe] = pca([Xon; Xoff]);
XonPC = Xon*C; XoffPC = Xoff*C;
figure; plot3(XonPC(:,1), XonPC(:,2), XonPC(:,3), 'o');
hold on; grid on; plot3(XoffPC(:,1), XoffPC(:,2), XoffPC(:,3), 'x');
xlabel('1'); ylabel('2'); zlabel('3');

%%
figure; plot(XonT(:,1), XonT(:,2), 'o');
hold on; grid on; plot(XoffT(:,1), XoffT(:,2), 'x');

%%
figure; plot(mean(Xon), 'b'); hold on; plot(mean(Xoff), 'r'); grid on;
plot(([-std(Xon); std(Xon)]+mean(Xon))', '--b');
plot(([-std(Xoff); std(Xoff)]+mean(Xoff))', '--r');
%figure; plot(mean(Son), 'b'); hold on; plot(mean(Soff), 'r'); grid on;
%plot(([-std(Son); std(Son)]+mean(Son))', '--b');
%plot(([-std(Soff); std(Soff)]+mean(Soff))', '--r');