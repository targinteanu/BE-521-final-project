

sub = 1; 
load('project_data_edit.mat')
X = train_ecog{sub}; Y = train_dg{sub};

%%
neur = 49; 
sig = X(:,neur);
sigmu = filter(Hbp, sig);

%%
feat = sqrt(movmean(sigmu.^2, 1000));

figure; clear ax; 
for f = 1:5
    ax(f) = subplot(6,1,f);
    plot(Y(:,f)); 
    grid on;
end
ax(6) = subplot(6,1,6); 
plot(feat); grid on;
linkaxes(ax, 'x'); 
clear ax;

%%
mu = filter(Hbp, X);
mu = sqrt(movmean(mu.^2, 1000));