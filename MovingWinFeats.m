function [Feats, righttime, lefttime] = MovingWinFeats(X, fs, winLen, winDisp, featFn)
% Applies a function (feature) to a signal in moving windows of specified 
% size and displacement.
% Inputs: 
%   x: raw signal vector 
%   fs: sampling rate in Hz
%   winLen: length of each window in seconds
%   winDisp: displacement between windows in seconds, measured from 
%       starting time of window n to starting time of window n+1
%   featFn: function handle for the desired feature 
% Note: appropriate window length and displacement must be chosen according 
% to sample rate so that window starting and ending indexes are real 
% samples, i.e. winLen*fs and winDisp*fs must be integers. 
% Outputs: 
%   feats: vector result of applying featFn to windowed x
%   righttime: time vector in seconds using right-aligned convention
%   lefttime: time vector in seconds using left-aligned convention
% Note: righttime and lefttime start at time 0. 

% convert winLen and winDisp from seconds to samples: 
winLen = winLen * fs; 
winDisp = winDisp * fs; 

[m,n] = size(X);
Feats = cell(1,n);

for col = 1:n
    x = X(:,col);

% find the indexes (samples) where each window should start 
startidx = 1:winDisp:m; % start of each window, in samples
% remove starting positions that would cause the window to exceed the signal:
startidx = startidx( startidx+winLen <= m );

% apply featFun to the values of x in each window to populate feats 
feats = arrayfun( @(i) featFn( x(i:(i+winLen)) ), startidx);

Feats{col} = feats';

end

Feats = cell2mat(Feats);

% set up the corresponding time vector 
winDisp = winDisp/fs; winLen = winLen/fs; 
righttime = ((1:length(feats)) - 1)*winDisp + winLen;
lefttime = ((1:length(feats)) - 1)*winDisp;

end