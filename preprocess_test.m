%load('project_data.mat');
Fs=1000;
winLen=0.100;%s
winDisp=0.05;%s
ave_time_domain=@(x) mean(x);

feature=cell(3,1);


for sub=1:3
    
    test_ecog_raw=test_ecog{sub};
    
    test_x=zeros(size(test_ecog_raw,1)/50-1,6*size(test_ecog_raw,2));
    
    for channel=1:size(test_ecog_raw,2)
        [fea1,ni,L,d]=MovingWinFeats(test_ecog_raw(:,channel), Fs, winLen, winDisp, ave_time_domain);
       % for five frequency bands:
        [fea2,ni,L,d]=MovingWinFeats_psd(test_ecog_raw(:,channel), Fs, winLen, winDisp, 5, 15);
        [fea3,ni,L,d]=MovingWinFeats_psd(test_ecog_raw(:,channel), Fs, winLen, winDisp, 20, 25);
        [fea4,ni,L,d]=MovingWinFeats_psd(test_ecog_raw(:,channel), Fs, winLen, winDisp, 75, 115);
        [fea5,ni,L,d]=MovingWinFeats_psd(test_ecog_raw(:,channel), Fs, winLen, winDisp, 125, 160);
        [fea6,ni,L,d]=MovingWinFeats_psd(test_ecog_raw(:,channel), Fs, winLen, winDisp, 160, 175);
        
        test_x(:,6*(channel-1)+1:6*channel)=[fea1',fea2',fea3',fea4',fea5',fea6'];

    end
    feature{sub}=test_x;
end
previewsWindow=4; %4 windows, lag time 200ms
x_test=cell(3,1);
for sub=1:3
    feature_raw=feature{sub};
    [m,n]=size(feature_raw);
    x_test_sub=zeros(size(test_ecog_raw,1)/50-previewsWindow,n*previewsWindow);
    
    for i=1:size(test_ecog_raw,1)/50-previewsWindow
        x_test_sub(i,:)=[feature_raw(i,:),feature_raw(i+1,:),feature_raw(i+2,:),feature_raw(i+3,:)];
    end
    
    x_test{sub}=x_test_sub;    
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
