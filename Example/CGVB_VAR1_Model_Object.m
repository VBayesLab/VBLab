% Example to fit a VAR(1) model, defined as a custom class object, using CGVB method
% We simulate a multivariate time-series  

clear 
clc

rng(2021)

% Setting
m = 2;   % Number of time series
T = 100; % Number of observations

% Generate data
y = randn(2,100);

% Create a VAR1 model object
Mdl = VAR1(m);

%% Run CGVB with defined model
Post_CGVB_VAR1 = CGVB(Mdl,y,...
                      'LearningRate',0.002,...       % Learning rate
                      'NumSample',50,...             % Number of samples to estimate gradient of lowerbound
                      'MaxPatience',20,...           % For Early stopping
                      'MaxIter',5000,...             % Maximum number of iterations
                      'GradWeight1',0.9,...          % Momentum 1
                      'GradWeight2',0.9,...          % Momentum 2
                      'WindowSize',10,...            % Smoothing window for lowerbound
                      'GradientMax',10,...           % For gradient clipping
                      'LBPlot',false); 
                 
                  
%% Plot variational distributions and lowerbound 
figure
% Extract variation mean and variance
mu_vb     = Post_CGVB_VAR1.Post.mu;
sigma2_vb = Post_CGVB_VAR1.Post.sigma2;

% Plot the variational distribution of each parameter
for i=1:Post_CGVB_VAR1.Model.NumParams
    subplot(2,4,i)
    vbayesPlot('Density',{'Normal',[mu_vb(i),sigma2_vb(i)]})
    grid on
    title(['\theta_',num2str(i)])
    set(gca,'FontSize',15)
end

% Plot the smoothed lower bound
subplot(2,4,7)
plot(Post_CGVB_VAR1.Post.LB_smooth,'LineWidth',2)
grid on
title('Lower bound')
set(gca,'FontSize',15)        

