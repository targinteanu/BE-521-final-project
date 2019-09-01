function [Xc, w, sparsity, fitstrength, epsilon, gwmags, wangles, Xdist] = ...
    YOLC(X, Y, epsilon, sigma, stepsize, wstart, discrete)

% perform Y-Optimized Linear Combination of variables X, i.e. LC of X that
% optimizes corresponding activity in Y. Reduces dimensionality of X to 1.
% YOLC first estimates the gradient dY/dX at each point X based on its
% relationship to neighboring points within a disc of size epsilon.
% Wherever there are insufficient neighbors, dY/dX is estimated based on a
% gaussian average of nearby values dependent on sigma. Then, it performs
% gradient ascent to determine the optimal weights, w, for the LC. 
% 
% Outputs: 
%   Xc: the optimal linear combination of X, i.e. X*w
%   w: the weights of the linear combination, corresponding to columns of X
%   sparsity: fraction of points that were filled in with nearby values
%       due to insufficient neighbors within epsilon. Larger sparsity 
%       indicates lower accuracy. 
%   fitstrength: average number of neighbors in epsilon (excluding points 
%       with insuficient neighbors) in relation to the number of neighbors
%       needed. Higher fitstrength indicates greater certainty. 
%   epsilon: size of disc, which may depend on mean and variance of X point
%       distances, depending on input preferences. 
%   gwmags: magnitude of gradient at each step of gradient ascent. When the
%       magnitude flattens, the optimum has likely been reached. 
%   wangles: at each step, angle between starting w and current w. When
%       this changes back and forth rapidly between steps, w is
%       oscillating, and the stepsize may be too large. 
% 
% Inputs: 
%   X: m-by-n input matrix where each column corresponds to one of n
%       variables and each row corresponds to one of m timepoints. 
%   Y: m-by-1 output column vector showing output at each of the m
%       timepoints. 
%   epsilon: radius of disc used to determine neighbors of each point, as
%       in the limit definition of a derivative. The gradient approximation
%       is most accurate for smaller epsilon, but depending on X, too small
%       may cause excess sparsity. Positive inputs will set the actual
%       value of the radius; negative inputs will specify how many standard
%       deviations below the average distance between points to set as the
%       disc radius. 
%   sigma: parameter used for gaussian estimation of derivative of points
%       with insufficient neighbors. Larger sigma will incorporate more
%       nearby points into the derivative and is generally more accurate.
%       Default = 20. 
%   stepsize: parameter used for gradient ascent of w. Larger step sizes
%       reach the optimum faster, but too large may overshoot, oscillate,
%       and become unstable. Default is set based on inputs and gradients. 
%   wstart: starting weight vector to use for gradient ascent. Default will
%       set weights to correspond with relative variance of X columns. 
%   discrete: false for continuous/analog Y, true for discrete Y (e.g.
%       classification) where each class has equal relationship to all
%       other classes. Default = false. 

if (nargin < 4)|(sigma == 0)
    sigma = 20; 
end
if nargin < 7
    discrete = false;
end

[m,n] = size(X);
SD = std(X);
if (nargin < 6)|(norm(wstart) == 0)
    wstart = (SD/norm(SD))';
end

Xdist = zeros(m,m);
for i = 1:m
    for j = 1:m
        Xdist(i,j) = norm(X(i,:) - X(j,:));
    end
end

if (epsilon <= 0)
    epsilon = mean(Xdist(:)) + epsilon*std(Xdist(:));
end
targetIdx = (Xdist <= epsilon) & Xdist;
numpts = sum(targetIdx,2);
sparsity = sum(numpts < n)/length(numpts);
fitstrength = mean(numpts(numpts >= n))/n;

dYdX = zeros(size(X));
for t = 1:m
    if numpts(t) >= n
        dX_t = X(targetIdx(t,:)',:) - X(t,:);
        dY_t = Y(targetIdx(t,:)') - Y(t);
        if discrete % only thing that matters is whether Y changed (dY =/= 0)
            dY_t = ~~dY_t;
        end
        dYdX_t = dX_t\dY_t;
        dYdX(t,:) = dYdX_t';
    else
        dYdX(t,:) = NaN; % leave a NaN hole where there aren't enough points
    end
end

% fill in NaN holes with surrounding data 
IdxSort = zeros(size(X)); IdxUnsort = zeros(size(X));
Xsorted = zeros(size(X)); 
dYdXsorted = zeros(size(dYdX));
for var = 1:n
    [Xsorted(:,var), IdxSort(:,var)] = sort(X(:,var));
    [~, IdxUnsort(:,var)] = sort(IdxSort(:,var));
    dYdXsorted(:,var) = dYdX(IdxSort(:,var),var);
end
dYdXsorted(isnan(dYdXsorted)) = 0;

% Determine filter length
filterExtent = ceil(4*sigma);
t = -filterExtent:filterExtent;
% Create 1-D Gaussian Kernel
c = 1/(sqrt(2*pi)*sigma);
gaussKernel = c * exp(-(t.^2)/(2*sigma^2));
% Normalize to ensure kernel sums to one
gaussKernel = gaussKernel/sum(gaussKernel);
dYdXsorted = imfilter(dYdXsorted, gaussKernel', 'conv', 'replicate');

dYdXfilt = zeros(size(dYdX));
for var = 1:n
    dYdXfilt(:,var) = dYdXsorted(IdxUnsort(:,var),var);
end
nanIdx = isnan(dYdX); 
dYdX(nanIdx) = dYdXfilt(nanIdx);

dYdXmag = sqrt(sum(dYdX.^2,2));
if (nargin < 5)|(stepsize == 0)
    %stepsize = 1e-3;
    stepsize = 100/(m*mean(dYdXmag)^2);
end
numsteps = ceil(1/stepsize);

w = wstart;
gwmags = zeros(1,numsteps); 
wangles = zeros(size(gwmags));
for i = 1:length(gwmags)
    gw = getgrad(w, dYdX);
    w = w + stepsize*gw; 
    w = w/norm(w);
    gwmags(i) = norm(gw);
    wangles(i) = wstart'*w;
end
wangles = acos(wangles);

Xc = X*w;

    function grad = getgrad(w, dydx)
        
        dotp = dydx*w;
        integrd = dotp.*dydx;
        grad = (2*sum(integrd))';
        
    end

end