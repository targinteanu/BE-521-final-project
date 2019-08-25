%%{
load('XYdata.mat'); 
Yall = Yall(1:100:end,:); Yall = Yall(11:end,:);
Y = Yall(:,3);

%{
% test: artificial Y and X -----------------------------------------
%Ycoeffs = [20, 0, 10, -.01]; Ypowers = [1,1,1,1];
%Ycoeffs = [-.085, .00062]; Ypowers = [2, 4];
%Ycoeffs = [5, 5]; Ypowers = [2, 2];
Ycoeffs = [20 -10 .01 -3 -15 .01]; Ypowers = [1, 3, 5, 3, 1, 1];
%Ycoeffs = [3, 4, -5]; Ypowers = [3, 2, 1];
numvars = length(Ycoeffs);
x0 = 1:1000; X = zeros(1000,numvars);
for var = 1:numvars
    X(:,var) = rand*sin(rand*x0 + rand) + 10*rand*sin(rand*x0 + rand) ...
        + rand*sin(10*rand*x0 + rand) + 10*rand*sin(10*rand*x0 + rand);
end
Y = zeros(length(X),1); 
for var = 1:numvars
    Y = Y + Ycoeffs(var)*X(:,var).^Ypowers(var);
end
% end test ----------------------------------------------------------
%}

%%
[m,n] = size(X);

% weight by variance of each variable 
SD = std(X);
%X = X.*(SD/max(SD));

%{
IdxSort = zeros(size(X)); IdxUnsort = zeros(size(X));
Xsorted = zeros(size(X)); Ysorted = zeros(size(X));
dXsort = zeros(size(Xsorted)); dYsort = zeros(size(Ysorted));

for var = 1:n
    [Xsorted(:,var), IdxSort(:,var)] = sort(X(:,var));
    Ysorted(:,var) = Y(IdxSort(:,var));
    [~, IdxUnsort(:,var)] = sort(IdxSort(:,var));
    
    dXsort(:,var) = gradient(Xsorted(:,var)); 
    dYsort(:,var) = gradient(Ysorted(:,var));
end

%dXsort = Xsorted(2:end,:) - Xsorted(1:(end-1),:); dXsort = [dXsort(1,:); dXsort];
%dYsort = Ysorted(2:end,:) - Ysorted(1:(end-1),:); dYsort = [dYsort(1,:); dYsort];

%dYdXsort = dYsort./dXsort; 
%dYdXsort(isinf(dYdXsort)) = 0;
dX = zeros(size(dXsort)); dY = zeros(size(dYsort));

for var = 1:n
    dX(:,var) = dXsort(IdxUnsort(:,var),var);
    dY(:,var) = dYsort(IdxUnsort(:,var),var);
end

A = cell(n,n); 
for i = 1:n
    for j = 1:n
        A{i,j} = diag( dX(IdxSort(:,i), j) );
    end
end
A = cell2mat(A); 

Aint = cell(n,n);
for i = 1:n
    for j = 1:n
        Aint{i,j} = (i==j)*dX(IdxSort(:,i), j)';
    end
end
Aint = cell2mat(Aint);
Aint = [Aint, (Ysorted(end,:) - Ysorted(1,:))']; % integral condition 

A = [A, dYsort(:)];
A = [A; Aint];
nu = null(A);
size(nu)
%figure; plot(n)
B = rref(A);

dYdXsort = zeros(size(dYsort)); 
dYdXsort(:) = B(1:(end-n),end);
dYdX = zeros(size(dYdXsort));
for var = 1:n
    dYdX(:,var) = dYdXsort(IdxUnsort(:,var),var);
end
%}

%{
Xdisp = permute(X, [1 3 2]) - permute(X, [3 1 2]);
Xdist = sqrt(sum(Xdisp.^2, 3));
%}
Xdist = zeros(m,m);
for i = 1:m
    for j = 1:m
        Xdist(i,j) = norm(X(i,:) - X(j,:));
    end
end

epsilon = mean(Xdist(:)) - .25*std(Xdist(:))
targetIdx = (Xdist <= epsilon) & Xdist;
[source, target] = find(targetIdx); numpts = sum(targetIdx,2);
sparsity = sum(numpts < n)/length(numpts)
fitstrength = mean(numpts(numpts >= n))/n

dYdX = zeros(size(X));
for t = 1:m
    if numpts(t) >= n
        dX_t = X(targetIdx(t,:)',:) - X(t,:);
        dY_t = Y(targetIdx(t,:)') - Y(t);
        dYdX_t = dX_t\dY_t;
        dYdX(t,:) = dYdX_t';
    else
        dYdX(t,:) = NaN; % leave a NaN hole where there aren't enough close points to estimate gradient
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

sigma = 20; 
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

% test: overwrite dYdX ------------------------------------
%{
for var = 1:n
    dYdX_act(:,var) = Ycoeffs(var)*Ypowers(var)*X(:,var).^(Ypowers(var)-1);
end
%}

%{
figure; hold on; grid on; 
colr = 'rbmgcy';
%for var = 1:n
for var = 1:6
    plot(X(:,var), dYdX(:,var), ['.' colr(var)]); 
%    plot(X(:,var), dYdX_act(:,var), ['o' colr(var)]); 
    plot(X((numpts < n),var), dYdX((numpts < n),var), ['s' colr(var)]);
end
%}
% end test ------------------------------------------------

%}
%%
%%{
wstart = (SD/norm(SD))';
%stepsize = 1e-18;
stepsize = 1e-3;
w = wstart;
gwmags = zeros(1,1000); 
wangles = zeros(size(gwmags));
for i = 1:length(gwmags)
    gw = getgrad(w, dYdX);
    w = w + stepsize*gw; 
    w = w/norm(w);
    gwmags(i) = norm(gw);
    wangles(i) = wstart'*w;
end
%%{
%[w (Ycoeffs/norm(Ycoeffs))']
wangles = acos(wangles);
figure; 
subplot(2,1,1); plot(gwmags); ylabel('grad mag'); xlabel('steps'); grid on;
subplot(2,1,2); plot(wangles); ylabel('w direction change'); xlabel('steps'); grid on;
%%}

%%{
figure; 
%subplot(2,1,1); plot(X*w, 'LineWidth', 2); grid on; ylabel('X'); xlabel('time');  
subplot(2,1,1); plot(X*w); grid on; ylabel('X'); xlabel('time');  
%hold on; plot(X);
subplot(2,1,2); plot(Y); grid on; ylabel('Y'); xlabel('time');
%%}

figure; plot(X*w, Y, '.'); grid on; xlabel('X'); ylabel('Y'); 
hold on; 
[C,S] = pca(X); plot(S(:,1), Y, '.');
legend('proposed', 'first PC');
%{
figure; plot(X*w, Y, '*'); grid on; xlabel('X'); ylabel('Y'); 
hold on; 
for var = 1:n
    plot(X(:,var), Y, '.');
end

% compare with PCA -----------------------------------------
[C,S] = pca(X); 
figure; plot(S(:,1), Y, '*'); grid on; xlabel('X'); ylabel('Y'); 
hold on; 
for var = 1:n
    plot(X(:,var), Y, '.');
end
title('first PC')
% ----------------------------------------------------------
%}

function grad = getgrad(w, dydx)

dotp = dydx*w; 
integrd = dotp.*dydx; 
grad = (2*sum(integrd))';

end

%}