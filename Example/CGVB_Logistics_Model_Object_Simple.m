% Script to run the Example 3.4 using a built-in Logistic Regression class of
% the VBLab package

clear  % Clear all variables 
clc % Clean workspace

% Random seed to reproduce results 
rng(2020)

% Load the LabourForce dataset
labour = readData('LabourForce',...     % Dataset name
                  'Type','Matrix',...   % Store data as a 2D array (default)
                  'Intercept', true);   % Add column of intercept (default)

% Compute number of features
n_features = size(labour,2)-1;

% Create a Logistic Regression model object
Mdl = LogisticRegression(n_features,...
                         'Prior',{'Normal',[0,50]});
                          
%% Run Cholesky GVB with random initialization
Post_CGVB  = CGVB(Mdl,labour,...
                 'LearningRate',0.002,...  % Learning rate
                 'NumSample',50,...        % Number of samples to estimate gradient of lowerbound
                 'MaxPatience',20,...      % For Early stopping
                 'MaxIter',5000,...        % Maximum number of iterations
                 'GradWeight1',0.9,...     % Momentum weight 1
                 'GradWeight2',0.9,...     % Momentum weight 2
                 'WindowSize',10,...       % Smoothing window for lowerbound
                 'StepAdaptive',500,...    % For adaptive learning rate
                 'GradientMax',10,...      % For gradient clipping    
                 'LBPlot',false);          % Dont plot the lowerbound when finish
     
%% Plot variational distributions and lowerbound 
figure
% Extract variation mean and variance
mu_vb     = Post_CGVB.Post.mu;
sigma2_vb = Post_CGVB.Post.sigma2;

% Plot the variational distribution for the first 8 parameters
for i=1:n_features
    subplot(3,3,i)
    vbayesPlot('Density',{'Normal',[mu_vb(i),sigma2_vb(i)]})
    grid on
    title(['\theta_',num2str(i)])
    set(gca,'FontSize',15)
end

% Plot the smoothed lower bound
subplot(3,3,9)
plot(Post_CGVB.Post.LB_smooth,'LineWidth',2)
grid on
title('Lower bound')
set(gca,'FontSize',15)
             
             
             
             
             