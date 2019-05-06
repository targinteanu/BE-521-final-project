sub = 1;
Y = train_dg{sub};
Ymm = movmean(Y, ceil(length(Y)/100));
Yactive = Ymm > .1; 
Y0 = Y; %Y = double(Yactive) + 1;

finger = {'thumb', 'index', 'middle', 'ring', 'little'};
figure; clear ax;
for f = 1:5
    ax(f) = subplot(5,1,f);
    plot(Y(:,f), 'b'); hold on; grid on; 
    plot(Ymm(:,f), '--r'); 
    plot(Yactive(:,f)*max(Y(:,f)), 'k'); 
    title(finger{f}); 
    ylabel('finger flexion'); 
end
xlabel('sample #'); 
linkaxes(ax); clear ax;