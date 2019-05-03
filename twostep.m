% evaluate an ML algorithm using cross validation. The ML algorithm MLfunc
% takes in training X and Y (xtr and ytr) and testing X (xte) and returns
% testing Y. Cross validation is performed on the training data 10 times
% using build-in crossval and averaged. The correlation for each finger is
% displayed. Then, an example cross validation is plotted for each finger
% with the red line separating training data from testing data. 

KNNfunc = @(Xtrain, Ytrain, Xtest) knnclassify(Xtest, Xtrain, Ytrain, 20);

MLfunc = @(xtr,ytr,xte) SVMfunc(xtr,ytr,xte);

sub = 1;
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
X = X - mean(X, 2); X = X./std(X, [], 2);
%Y = Y - mean(Y, 2); Y = Y./std(Y, [], 2);

%{
sub = 1;
load('preprocessed_data.mat')
%load('project_data.mat')
%X = train_ecog{1}; Y = train_dg{1};
X = x_train{sub}; Y = y_train{sub};

[C,S,~,~,pe] = pca(X);
X0 = X; X = S(:,1:100);
%}

addpath(genpath('C:\Users\Toren\Desktop\BE521final\libsvmmatlab'))
savepath

tic 
crossvalrho = arrayfun(@(f) mean(...
    crossval(@(Xtr,Ytr,Xte,Yte) ...
        sum(Yte == MLfunc(Xtr,Ytr,Xte))/length(Yte), ...
        X, Y(:,f))), 1:5)
toc
%}

nTR = floor(length(Y(:,1))*.5);
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
    acc = sum(Ytestpred(:,f) == YTE(:,f))/length(YTE(:,f));
    title(['test acc. = ' num2str(acc) ' | crossval ' num2str(crossvalrho(f))]);
    ylim([.5 2.5]);
end
linkaxes(ax); clear ax;

function Ypred = lassofunc(xtrain, ytrain, xtest)

        [W,FitInfo] = lasso(xtrain,ytrain, 'lambda', 0.01);
        coef0 = FitInfo.Intercept;
        Ypred=xtest*W(:)+coef0;
        
end

function Ypred = SVMfunc(Xtrain, Ytrain, Xtest)

% balance class sizes in training set
class = [1, 2];
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

    logreg = mnrfit(Xtrain, Ytrain);
    testProb = mnrval(logreg, Xtest);
    [~,Ypred] = max(testProb, [], 2);
    
end