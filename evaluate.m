% evaluate an ML algorithm using cross validation. The ML algorithm MLfunc
% takes in training X and Y (xtr and ytr) and testing X (xte) and returns
% testing Y. Cross validation is performed on the training data 10 times
% using build-in crossval and averaged. The correlation for each finger is
% displayed. Then, an example cross validation is plotted for each finger
% with the red line separating training data from testing data. 

sub = 1; 
%load('preprocessed_data_edit.mat')
X = x_train{sub}; Y = y_train{sub};

addpath(genpath('C:\Users\Toren\Desktop\BE521final\libsvmmatlab'))
savepath

MLfunc = @(xtr,ytr,xte) lassofunc(xtr,ytr,xte);

crossvalrho = arrayfun(@(f) mean(...
    crossval(@(Xtr,Ytr,Xte,Yte) ...
        corr(Yte, MLfunc(Xtr,Ytr,Xte)), ...
        X, Y(:,f))), 1:5)

nTR = floor(length(Y(:,1))*.9);
YTR = Y(1:nTR,:); YTE = Y((nTR+1):end,:);
XTR = X(1:nTR,:); XTE = X((nTR+1):end,:);
Ytrainpred = zeros(size(YTR));
figure; clear ax;
for f = 1:5
    Ytrainpred(:,f) = MLfunc(XTR, YTR(:,f), XTR);
    Ytestpred(:,f) = MLfunc(XTR, YTR(:,f), XTE);
    ax(f) = subplot(5,1,f);
    plot(Y(:,f), 'k'); hold on; plot([Ytrainpred(:,f); Ytestpred(:,f)], 'b');
    plot([nTR nTR], [min(Y(:)) max(Y(:))], 'r');
    grid on; 
    title(['avg. \rho from crossval = ' num2str(crossvalrho(f))]);
end
linkaxes(ax); clear ax;

function Ypred = lassofunc(xtrain, ytrain, xtest)

        [W,FitInfo] = lasso(xtrain,ytrain, 'lambda', 0.01);
        coef0 = FitInfo.Intercept;
        Ypred=xtest*W(:)+coef0;
        
end

function Ypred = SVMfunc(Xtrain, Ytrain, Xtest)

    mySVM = svmtrain(Ytrain, Xtrain); 
    Ypred = svmpredict(zeros(length(Xtest(:,1)), length(Ytrain(1,:))),...
        Xtest, mySVM);
    
end