% Script to fit a Logistic Regression model, defined as a class object,
% using the VAFC method

clear  % Clear all variables 
clc % Clean workspace

% Random seed to reproduce results 
rng(2020)

% Load GermanCredit data. 
credit = readData('GermanCredit',...    % Dataset name
                  'Type','Matrix',...   % Store data as a 2D array (default)
                  'Intercept', true);   % Add column of intercept (default)

% Compute number of features
n_features = size(credit,2)-1;

% Create a Logistic Regression model object
Mdl = LogisticRegression(n_features);
                          
%% Run Cholesky GVB with random initialization
Post_VAFC = VAFC(Mdl,credit,...
                 'NumFactor',10, ...      % Number of factors of the loading matrix
                 'NumSample',50,...       % Number of samples to estimate gradient of lowerbound
                 'LearningRate',0.05,...  % Learning rate
                 'MaxPatience',20,...     % For Early stopping
                 'MaxIter',10000,...      % Maximum number of iterations
                 'GradientMax',200,...    % For gradient clipping    
                 'WindowSize',5, ...      % Smoothing window for lowerbound
                 'LBPlot',false);         % Dont plot the lowerbound when finish             

%% Plot variational distributions and lowerbound 
figure
% Extract variation mean and variance
mu_vb     = Post_VAFC.Post.mu;
sigma2_vb = Post_VAFC.Post.sigma2;

% Plot the variational distribution for the first 8 parameters
for i=1:8
    subplot(3,3,i)
    vbayesPlot('Density',{'Normal',[mu_vb(i),sigma2_vb(i)]})
    grid on
    title(['\theta_',num2str(i)])
    set(gca,'FontSize',15)
end

% Plot the smoothed lower bound
subplot(3,3,9)
plot(Post_VAFC.Post.LB_smooth,'LineWidth',2)
grid on
title('Lower bound')
set(gca,'FontSize',15)

