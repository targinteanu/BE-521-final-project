load('project_data_edit.mat');

%% window features (train)

Fs=1000;
winLen=0.1;%s
winDisp=0.05;%s
ave_time_domain=@(x) mean(x);

feature=cell(3,1);


for sub=1:3
    
    train_ecog_raw=train_ecog{sub};
    
    train_x=zeros(5999,6*size(train_ecog_raw,2));
    
    for channel=1:size(train_ecog_raw,2)
        [fea1,ni,L,d]=MovingWinFeats(train_ecog_raw(:,channel), Fs, winLen, winDisp, ave_time_domain);
       % for 5 frequency bands:
        [fea2,ni,L,d]=MovingWinFeats_psd(train_ecog_raw(:,channel), Fs, winLen, winDisp, 5, 15);
        [fea3,ni,L,d]=MovingWinFeats_psd(train_ecog_raw(:,channel), Fs, winLen, winDisp, 20, 25);
        [fea4,ni,L,d]=MovingWinFeats_psd(train_ecog_raw(:,channel), Fs, winLen, winDisp, 75, 115);
        [fea5,ni,L,d]=MovingWinFeats_psd(train_ecog_raw(:,channel), Fs, winLen, winDisp, 125, 160);
        [fea6,ni,L,d]=MovingWinFeats_psd(train_ecog_raw(:,channel), Fs, winLen, winDisp, 160, 175);
        
        train_x(:,6*(channel-1)+1:6*channel)=[fea1',fea2',fea3',fea4',fea5',fea6'];

    end
    feature{sub}=train_x;
end


%% downsampled Y
previewsWindow=4;% use previous 4 windows 
y_train=cell(3,1);

for sub=1:3
    y_sub=zeros(6000-previewsWindow,5);
    train_dg_raw=train_dg{sub};
    for finger=1:5
        %plot(train_dg_raw(:,finger),'.')
        %y=train_dg_raw(50*(0:length(y)-1)+1,finger);
%y=decimate(train_dg_raw(:,finger),50);  %filter?
y=decimate( movmean(train_dg_raw(:,finger), 100), 50);
         
    % x=50*(0:length(y)-1);
        y_sub(:,finger)=y(previewsWindow+1:end); % predict fourth window using 1 2 and 3 windows
    end
    y_train{sub}=y_sub; 
end
%save('xy.mat','x_train','y_train','x_test')

% x=50*(0:length(y)-1)+50;

% xx=0:300000-1;
% yy = spline(x,y,xx); % cubic spline data interpolation.
%        hold on;
% plot(x,y,'*')
% plot(yy,'.')
% corr(train_dg_raw(:,finger),yy')


%% x_train
x_train=cell(3,1);
for sub=1:3
    feature_raw=feature{sub};
    [m,n]=size(feature_raw);
    x_train_sub=zeros(6000-previewsWindow,n*previewsWindow);
    
    for i=1:6000-previewsWindow
        x_train_sub(i,:)=[feature_raw(i,:),feature_raw(i+1,:),feature_raw(i+2,:),feature_raw(i+3,:)];% feature matrix, 3 windows together
    end
    
    x_train{sub}=x_train_sub;    
end


%save('xy.mat','x_train','y_train','x_test')




%% function

function [feature,ni,L,d]=MovingWinFeats(x, fs, winLen, winDisp, featFn)

    L=winLen*fs;
    d=winDisp*fs;
    ni=mod(numel(x)-L,d);
    NumOfWin=(numel(x)-(ni+(L-d)))/(d);
    feature=zeros(1,NumOfWin);
    for i=1:NumOfWin
        feature(i)=featFn(x(ni+(i-1)*d+1:ni+(i-1)*d+L));
    end
    
end

function feat=average_freq(x, min, max, Fs)
    
    N = length(x);
    xdft = fft(x);
    xdft = xdft(1:N/2+1);
    psdx = (1/(Fs*N)) * abs(xdft).^2;
    psdx(2:end-1) = 2*psdx(2:end-1);
    freq = 0:Fs/length(x):Fs/2;

    feat=mean(psdx(freq<=max & freq>=min));
    
end


function [feature,ni,L,d]=MovingWinFeats_psd(x, fs, winLen, winDisp, min, max)

    L=winLen*fs;
    d=winDisp*fs;
    ni=mod(numel(x)-L,d);
    NumOfWin=(numel(x)-(ni+(L-d)))/(d);
    feature=zeros(1,NumOfWin);
    for i=1:NumOfWin
        feature(i)=average_freq(x(ni+(i-1)*d+1:ni+(i-1)*d+L), min, max,fs);
    end
    
end



