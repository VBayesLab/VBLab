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
Estmdl_1  = CGVB(Mdl,labour,...
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
     
%% Run Cholesky GVB with MLE initialization
% Random seed to reproduce results 
rng(2020)

theta_init = Mdl.initParams('MLE',labour); 
Estmdl_2  = CGVB(Mdl,labour,...
                'MeanInit',theta_init,... % 
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

%% Then compare convergence of lowerbound in 2 cases 
figure
hold on
grid on
plot(Estmdl_1.Post.LB_smooth,'-r','LineWidth',2)
plot(Estmdl_2.Post.LB_smooth,'--b','LineWidth',2)
title('Lowerbound')
legend('Random Initialization','MLE Initialization' )

%% It is useful to compare the approximate posterior density to the true density obtain by MCMC
Post_MCMC = MCMC(Mdl,labour,...
                 'NumMCMC',50000,...
                 'ParamsInit',theta_init,...
                 'Verbose',1);
             
%% Compare densities by CGVB and MCMC
% Get posterior mean and trace plot for a parameter to check the mixing 
[mcmc_mean,mcmc_std,mcmc_chain] = Post_MCMC.getParamsMean('BurnInRate',0.4,...
                                                          'PlotTrace',1);

% Plot density
fontsize  = 20;
numparams = Estmdl_2.Model.NumParams;

% Extract variation mean and variance
mu_vb     = Estmdl_2.Post.mu;
sigma2_vb = Estmdl_2.Post.sigma2;

figure
for i = 1:numparams
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
plot(Estmdl_2.Post.LB_smooth,'LineWidth',1.5)
title('Lower bound','FontSize', fontsize)
             
             
             
             
             