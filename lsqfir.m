function [w, yfilt] = lsqfir(x, y, L)

x = x';
X = zeros(length(x)-L, L); 
for i = 1:(length(x)-L)
    X(i,:) = x(i:(i+L-1));
end

w = (X'*X)\X'*y(1:(end-L));
yfilt = y;
for i = (L+1):length(y)
    yfilt(i) = y((i-L):(i-1))'*w;
end
yfilt = [yfilt((L+1):end); yfilt(end)*ones(L,1)]; 

end