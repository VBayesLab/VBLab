% Script to fit a Logistic Regression model, defined as a class object,
% using the NAGVAC method

clear  % Clear all variables 
clc % Clean workspace

% Random seed to reproduce results 
rng(2020)

% Load LabourForce data. 
labour = readData('LabourForce',...     % Dataset name
                  'Type','Matrix',...   % Store data as a 2D array (default)
                  'Intercept', true);   % Add column of intercept (default)

% Compute number of features
n_features = size(labour,2)-1;

% Create a Logistic Regression model object
Mdl = LogisticRegression(n_features,...
					    'Prior',{'Normal',[0,50]});
                          
%% Run Cholesky GVB with random initialization
theta_init = Mdl.initParams('MLE',labour); 
Post_VAFC = NAGVAC(Mdl,labour,...
                  'MeanInit',theta_init,... % Initial values of variational mean
                  'NumSample',200,...       % Number of samples to estimate gradient of lowerbound
                  'LearningRate',0.001,...  % Learning rate
                  'MaxPatience',20,...      % For Early stopping
                  'MaxIter',10000,...       % Maximum number of iterations
                  'GradientMax',200,...     % For gradient clipping    
                  'WindowSize',30, ...      % Smoothing window for lowerbound
                  'LBPlot',true);           % Dont plot the lowerbound when finish
             

%% It is useful to compare the approximate posterior density to the true density obtain by MCMC
Post_MCMC = MCMC(Mdl,labour,...
                 'NumMCMC',50000,...
                 'ParamsInit',theta_init,...
                 'Verbose',1000);       % Display after each 1000 iterations
             
%% Compare densities by CGVB and MCMC
% Get posterior mean and trace plot for a parameter to check the mixing 
[mcmc_mean,mcmc_std,mcmc_chain] = Post_MCMC.getParamsMean('BurnInRate',0.5,...
                                                          'PlotTrace',1);

% Plot density
fontsize  = 20;

% Extract variation mean and variance
mu_vb     = Post_VAFC.Post.mu;
sigma2_vb = Post_VAFC.Post.sigma2;

% Plot only variational distribution for 8 parameters 
figure
for i = 1:8 
    subplot(3,3,i)
    xx = mcmc_mean(i)-4*mcmc_std(i):0.002:mcmc_mean(i)+4*mcmc_std(i);
    yy_mcmc = ksdensity(mcmc_chain(:,i),xx);    
    yy_vb = normpdf(xx,mu_vb(i),sqrt(sigma2_vb(i)));
    plot(xx,yy_mcmc,'-k',xx,yy_vb,'--b','LineWidth',1.5)
    line([theta_init(i) theta_init(i)],ylim,'LineWidth',1.5,'Color','r')    
    str = ['\theta_', num2str(i)];   
    title(str,'FontSize', fontsize)
    legend('MCMC','VB')
end
subplot(3,3,9)
plot(Post_VAFC.Post.LB_smooth,'LineWidth',1.5)
title('Lower bound','FontSize', fontsize)          
              