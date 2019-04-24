%0.39 correlation. (fixed small syntax errors from 0.40)
% load('xy.mat')
% load('project_data')

predicted_dg=cell(3,1);

for sub=1:3
    
    xtrain=x_train{sub};%feature matrix 
    %[m,n]=size(xtrain);
    ytrain_all=y_train{sub};
    xtest=x_test{sub};
    [m,n]=size(xtest);
    ypred_all=zeros(147500,5);
    
    for finger=1:5
        ytrain=ytrain_all(:,finger); 
        % lambda is a regularization coefficient
        [W,FitInfo] = lasso(xtrain,ytrain, 'lambda', 0.01);
        coef0 = FitInfo.Intercept;
        pred=xtest*W(:)+coef0;
%       train_predict=corr(train_predict), ytrain);
%         plot(ytrain);
%         hold on
%         plot(train_predict);
        %% interpolated the prediction
         x=50*(0:numel(pred)+previewsWindow-1)+50; %to check where y is 
         y=[pred(1);pred(1);pred(1);pred(1);  pred];% how to predict first 3 data points? Current repeating the first prediction (which should be the forth data points)
         xx=1:147500;
         ypred_all(:,finger) = (spline(x,y,xx))'; 
 %         plot(x,y,'*')
 %         hold on;
 %         plot(xx,yy,'.')       
    end
    predicted_dg{sub}=ypred_all; 
%   rho_malika=corr(predicted_dg{sub}, ypred_all);
end
save ('predicted_dg.mat','predicted_dg');

