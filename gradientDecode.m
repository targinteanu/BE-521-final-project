%load('XYdata.mat'); 
%Yall = Yall(1:100:end,:); Yall = Yall(11:end,:);
%Y = Yall(:,3);

% test: artificial Y and X -----------------------------------------
%Ycoeffs = [20, 0, 10, -.01, -40]; Ypowers = [1,1,1,1,1];
Ycoeffs = [-.085, .00062]; Ypowers = [2, 4];
%Ycoeffs = [20 -10 .01 -3 -15 .01]; Ypowers = [3, 3, 3, 3, 3];
numvars = length(Ycoeffs);
x0 = 1:1000; X = zeros(1000,numvars);
for var = 1:numvars
    X(:,var) = rand*sin(rand*x0 + rand) + 10*rand*sin(rand*x0 + rand) ...
        + rand*sin(10*rand*x0 + rand) + 10*rand*sin(10*rand*x0 + rand);
end
%Y = Ycoeffs(1)*X(:,1) + Ycoeffs(2)*X(:,2) + Ycoeffs(3)*X(:,3) ...
%    + Ycoeffs(4)*X(:,4) + Ycoeffs(5)*X(:,5);
Y = zeros(length(X),1); 
for var = 1:numvars
    Y = Y + Ycoeffs(var)*X(:,var).^Ypowers(var);
end
% end test ----------------------------------------------------------

[m,n] = size(X);

% weight by variance of each variable 
SD = std(X);
%X = X.*(SD/max(SD));

IdxSort = zeros(size(X)); IdxUnsort = zeros(size(X));
Xsorted = zeros(size(X)); Ysorted = zeros(size(X));

for var = 1:n
    [Xsorted(:,var), IdxSort(:,var)] = sort(X(:,var));
    Ysorted(:,var) = Y(IdxSort(:,var));
    [~, IdxUnsort(:,var)] = sort(IdxSort(:,var));
end

dXsort = Xsorted(2:end,:) - Xsorted(1:(end-1),:); dXsort = [dXsort(1,:); dXsort];
dYsort = Ysorted(2:end,:) - Ysorted(1:(end-1),:); dYsort = [dYsort(1,:); dYsort];

dYdXsort = dYsort./dXsort; 
dYdXsort(isinf(dYdXsort)) = 0;
dYdX = zeros(size(dYdXsort));

for var = 1:n
    dYdX(:,var) = dYdXsort(IdxUnsort(:,var),var);
end

% test: overwrite dYdX ------------------------------------
for var = 1:n
    dYdX(:,var) = Ycoeffs(var)*X(:,var).^(Ypowers(var)-1);
end
% end test ------------------------------------------------

wstart = (SD/norm(SD))';
%stepsize = 1e-18;
stepsize = 1e-5;
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
[w (Ycoeffs/norm(Ycoeffs))']
wangles = acos(wangles);
figure; 
subplot(2,1,1); plot(gwmags); ylabel('grad mag'); xlabel('steps'); grid on;
subplot(2,1,2); plot(wangles); ylabel('w direction change'); xlabel('steps'); grid on;
%}

%%{
figure; 
subplot(2,1,1); plot(X*w, 'LineWidth', 2); grid on; ylabel('X'); xlabel('time');  
hold on; plot(X);
subplot(2,1,2); plot(Y); grid on; ylabel('Y'); xlabel('time');
%}

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
% ----------------------------------------------------------

function grad = getgrad(w, dydx)

dotp = dydx*w; 
integrd = dotp.*dydx; 
grad = (2*sum(integrd))';

end