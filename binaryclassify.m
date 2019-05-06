% code to binary classify whether the finger is moving or not based on mu
% wave (7.5-12.5 Hz band) amplitude. Tested on subject 1 with SVM, KNN, and
% logistic regression. Not successful. 

sub = 1; 
load('project_data.mat')
X = train_ecog{sub}; Y = train_dg{sub}; % unprocessed input/output data
load('mufilter.mat') % bandpass filter (Hbp) that gives mu wave band 

%% preprocess X (the signal from each neuron) into a feature matrix

mu_raw = filter(Hbp, X); % get only the signal in the mu range 

% construct a feature matrix MU. Features at time t are the average mu wave 
% amplitudes in the previous <numwins> windows for all neurons. 
MU = mu_raw;
winsize = 1000; % ms
windisp = 100; % ms
numwins = 10;
MU = sqrt(movmean(MU.^2, winsize)); % RMS ~ signal amplitude 
MU2 = MU(1:windisp:end,:); % turn into windows 
MU3 = zeros( size(MU2).*[1,numwins] - [numwins,0] );
for t = 1:length(MU3)
    wins = MU2(t:(t+numwins-1),:);
    MU3(t,:) = wins(:)';
end
MU = MU3; 

X = MU; % set X as the feature matrix
%X = X - mean(X, 2); X = X./std(X, [], 2); % normalize X (tried with and without this)
X = X - mean(X);

%% reduce dimensionality 
[C,S,~,~,pe] = pca(X);
X = S(:,1:50);

%% preprocess Y into two categories (finger moving or not) 

Ymm = movmean(Y, ceil(length(Y)/100)); % smooth finger wiggling into one long movement
Yactive = Ymm > .1; % threshold into two categories 
Y = double(Yactive) + 1; % 1 = not moving, 2 = moving

% downsample and trim Y to match size of X
ds = floor(length(Y)/length(X));
Y = Y(1:ds:end,:);
trim = length(Y)-length(X) + 1;
Y = Y(trim:end,:);

%% run and evaluate a binary classification 

KNNfunc = @(Xtrain, Ytrain, Xtest) knnclassify(Xtest, Xtrain, Ytrain, 20);
% supporting function for if you want to use KNN

% ** important: change MLfunc to KNNfunc, SVMfunc, or logregfunc depending
%    on which one you want to use ** 
MLfunc = @(xtr,ytr,xte) KNNfunc(xtr,ytr,xte);

% ** important: include library for libsvm if you want to use it 

% compute cross validation on 10 randomized testing/training sets 
%{
tic 
crossvalrho = arrayfun(@(f) mean(...
    crossval(@(Xtr,Ytr,Xte,Yte) ...
        sum(Yte == MLfunc(Xtr,Ytr,Xte))/length(Yte), ...
        X, Y(:,f))), 1:5)
toc
%}

%% display an example of a testing and training set with predictions 
% for each finger, display the actual Y in black and the predicted Y in
% blue. The red vertical line separates testing from training sets. 

fraction_training = .5; % fraction of the data to use for training 
nTR = floor(length(Y(:,1))*fraction_training);
YTR = Y(1:nTR,:); YTE = Y((nTR+1):end,:);
XTR = X(1:nTR,:); XTE = X((nTR+1):end,:);
Ytrainpred = zeros(size(YTR)); Ytestpred = zeros(size(YTE));
figure; clear ax;
for f = 1:5
    Ytrainpred(:,f) = MLfunc(XTR, YTR(:,f), XTR);
    Ytestpred(:,f) = MLfunc(XTR, YTR(:,f), XTE);
    ax(f) = subplot(5,1,f);
    plot(Y(:,f), 'k'); hold on; plot([Ytrainpred(:,f); Ytestpred(:,f)], '--b');
    plot([nTR nTR], [min(Y(:)) max(Y(:))], 'r');
    grid on; 
    testacc = sum(Ytestpred(:,f) == YTE(:,f))/length(YTE(:,f));
    trainacc = sum(Ytrainpred(:,f) == YTR(:,f))/length(YTR(:,f));
    title(['test acc. = ' num2str(testacc) ...
        ' | train ' num2str(trainacc)]);% ...
%        ' | crossval ' num2str(crossvalrho(f))]);
    ylim([.5 2.5]);
end
linkaxes(ax); clear ax;

%% supporting functions 
function Ypred = lassofunc(xtrain, ytrain, xtest)
% LASSO

        [W,FitInfo] = lasso(xtrain,ytrain, 'lambda', 0.01);
        coef0 = FitInfo.Intercept;
        Ypred=xtest*W(:)+coef0;
        
end

function Ypred = SVMfunc(Xtrain, Ytrain, Xtest)
% SVM

% balance class sizes in training set (tried with and without this) 
class = [1, 2]; % 1 = not moving, 2 = moving 
classidx = cell(size(class)); classcount = zeros(size(class));
for i = 1:length(class)
    classidx{i} = find(Ytrain == class(i));
    classcount(i) = length(classidx{i});
end
cutoff = min(classcount); 
Xtr = zeros(cutoff, length(Xtrain(1,:))); Ytr = zeros(cutoff, 1);
for i = 1:length(class)
    idx = classidx{i}(1:cutoff);
    pos = ((i-1)*cutoff+1);
    Xtr(pos:(pos+cutoff-1),:) = Xtrain(idx,:);
    Ytr(pos:(pos+cutoff-1)) = Ytrain(idx);
end

% Run the SVM. The top version uses built-in MATLAB SVM. The bottom
% (currently commented) uses the libsvm provided on canvas. I tried both. 
    %%{
    mySVM = fitcsvm(Xtr, Ytr, 'KernelFunction', 'gaussian');
    Ypred = mySVM.predict(Xtest);
    %}
    %{
    mySVM = svmtrain(Ytrain, Xtrain); 
    Ypred = svmpredict(zeros(length(Xtest(:,1)), length(Ytrain(1,:))),...
        Xtest, mySVM);
    %}
    
end

function Ypred = logregfunc(Xtrain, Ytrain, Xtest)
% logistic regression 

    logreg = mnrfit(Xtrain, Ytrain);
    testProb = mnrval(logreg, Xtest);
    [~,Ypred] = max(testProb, [], 2);
    
end