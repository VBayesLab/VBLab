classdef MCMC < handle & matlab.mixin.CustomDisplay
    %MCMC Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Method
        Model           % Instance of model to be fitted
        ModelToFit      % Name of model to be fitted
        SeriesLength    % Length of the series    
        NumMCMC         % Number of MCMC iterations 
        BurnInRate      % Percentage of sample for burnin
        BurnIn          % Number of samples for burnin
        TargetAccept    % Target acceptance rate
        NumCovariance   % Number of latest samples to calculate adaptive covariance matrix for random-walk proposal
        SaveFileName    % Save file name
        SaveAfter       % Save the current results after each 5000 iteration
        ParamsInit      % Initial values of model parameters
        Seed            % Random seed
        Post            % Struct to store estimation results
        Initialize      % Initialization method
        LogLikelihood   % Handle of the log-likelihood function
        PrintMessage    % Custom message during the sampling phase
        CPU             % Sampling time    
        Verbose         % Turn on of off printed message during sampling phase
        SigScale
        Scale
        Params
    end
    
    methods
        function obj = MCMC(model,data,varargin)
            %MCMC Construct an instance of this class
            %   Detailed explanation goes here
            obj.Method        = 'MCMC';
            obj.Model         = model;   
            obj.ModelToFit    = model.ModelName;
            obj.NumMCMC       = 50000;
            obj.TargetAccept  = 0.25;
            obj.BurnInRate    = 0.2;
            obj.NumCovariance = 2000;
            obj.SigScale      = 0.01;
            obj.Scale         = 1;
            obj.SaveAfter     = 0;
            obj.Verbose       = 100;
            obj.ParamsInit    = [];
            
            if nargin > 2
                %Parse additional options
                paramNames = {'NumMCMC'         'BurnInRate'      'TargetAccept'      'NumCovariance'   ...
                              'ParamsInit'      'SaveFileName'    'SaveAfter'         'Verbose'     ...
                              'Seed'            'SigScale'        'Scale'};
                paramDflts = {obj.NumMCMC       obj.BurnInRate    obj.TargetAccept    obj.NumCovariance  ...
                              obj.ParamsInit    obj.SaveFileName  obj.SaveAfter       obj.Verbose   ...
                              obj.Seed          obj.SigScale     obj.Scale};

                [obj.NumMCMC,...
                 obj.BurnInRate,...
                 obj.TargetAccept,...
                 obj.NumCovariance,...
                 obj.ParamsInit,...
                 obj.SaveFileName,...
                 obj.SaveAfter,...
                 obj.Verbose,...
                 obj.Seed,...
                 obj.SigScale,...
                 obj.Scale] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
            end
            
            obj.BurnIn = floor(obj.BurnInRate*obj.NumMCMC);
            
            % Set up saved file name
            DateVector = datevec(date);
            [~, MonthString] = month(date);
            date_time = ['_',num2str(DateVector(3)),'_',MonthString,'_'];
            obj.SaveFileName = ['Results_MCMC',date_time];
            
            % Run MCMC
            obj.Post   = obj.fit(data); 
            
        end
        
        % Sample a posterior using MCMC
        function Post = fit(obj,data)
            
            % Extract sampling setting
            model        = obj.Model;
            num_params   = model.NumParams;
            verbose      = obj.Verbose;
            numMCMC      = obj.NumMCMC;
            scale        = obj.Scale;
            V            = obj.SigScale*eye(num_params);
            accept_rate  = obj.TargetAccept;
            N_corr       = obj.NumCovariance;
            saveAfter    = obj.SaveAfter;
            saveFileName = obj.SaveFileName;
            params_init  = obj.ParamsInit;
            
            thetasave    = zeros(numMCMC,num_params);
                         
            % Get initial values of parameters
            if ~isempty(params_init) % If a vector of initial values if provided
                if (length(params_init) ~= num_params)
                    error(utils_errorMsg('vbayeslab:InitVectorMisMatched'))
                else
                    params = params_init;
                end
            else
                params = model.initParams('Prior');
            end
            
            % Make sure params is a row vector
            params = reshape(params,1,num_params);
            
            % For the first iteration
            log_prior = model.logPriors(params);
            lik       = model.logLik(data,params);
            jac       = model.logJac(params);
            post      = log_prior + lik;
            
            tic
            for i=1:numMCMC
                if(verbose)
                    if(mod(i,verbose)==0)
                        disp(['iter: ',num2str(i),'(',num2str(i/numMCMC*100),'%)'])
                    end
                end

                % Transform params to normal distribution scale
                params_normal = model.toNormalParams(params);
                
                % Using multivariate normal distribution as proposal distribution
                sample = mvnrnd(params_normal,scale.*V);

                % Convert theta to original distribution
                theta = model.toOriginalParams(sample);

                % Calculate acceptance probability for new proposed sample
                log_prior_star = model.logPriors(theta);
                lik_star       = model.logLik(data,theta);
                jac_star       = model.logJac(theta);
                post_star      = log_prior_star + lik_star;

                A = rand();
                r = exp(post_star-post+jac-jac_star);
                C = min(1,r);   
                if A<=C
                    params = theta;
                    post   = post_star;
                    jac    = jac_star;
                end
                thetasave(i,:) = params;

                % Adaptive scale for proposal distribution
                if i > 50
                    scale = utils_update_sigma(scale,C,accept_rate,i,num_params);
                    if (i > N_corr)
                        V = cov(thetasave(i-N_corr+1:i,:));
                    else
                        V = cov(thetasave(1:i,:));
                    end
                    V = utils_jitChol(V);
                end
                Post.theta(i,:) = params;
                Post.scale(i)   = scale;

                % Store results after each 5000 iteration
                if(saveAfter>0)
                    if mod(i,saveAfter)==0
                        save(saveFileName,'Post')
                    end
                end
            end
            Post.cpu = toc; 
        end
        
        % Function to get parameter means given MCMC samples
        function [params_mean,params_std,params] = getParamsMean(obj,varargin)
            post = obj.Post;
            burnin = [];
            burninrate = [];
            PlotTrace = [];   % Array of indexes of model parameters
            subplotsize = [];
            if nargin > 0
                %Parse additional options
                paramNames = {'BurnIn'          'BurnInRate'  'PlotTrace'   'SubPlot'};
                paramDflts = {burnin             burninrate    PlotTrace    subplotsize};

                [burnin,...
                 burninrate,...
                 PlotTrace,...
                 subplotsize] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
            end
            
            if(isempty(burnin))
                burnin = obj.BurnIn;
            end
            
            if(isempty(burninrate))
                burninrate = obj.BurnInRate;
            else
                burnin = floor(burninrate*obj.NumMCMC);
            end
            
            params_mean = mean(post.theta(burnin+1:end,:));
            params_std  = sqrt(mean(post.theta(burnin+1:end,:).^2)-params_mean.^2);
            params      = post.theta(burnin+1:end,:);
            % If user wants to plot trace of the first parameter to check
            % the mixing
            if (~isempty(PlotTrace) && ~isempty(subplotsize))
                nrow = subplotsize(1);
                ncol = subplotsize(2);
                
                figure
                for i=1:length(PlotTrace)
                    subplot(nrow,ncol,i)
                    plot(post.theta(burnin+1:end,PlotTrace(i)))
                    title(['\theta_',num2str(i)],'FontSize', 20)
                end
            end
        end
    end
end

