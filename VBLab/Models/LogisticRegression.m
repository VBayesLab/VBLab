classdef LogisticRegression
    %LOGISTICREGRESSION Summary of this class goes here
    %   Detailed explanation goes here
    % Attributes
    properties 
        ModelName      % Model name 
        NumParams      % Number of parameters
        PriorInput     % Prior specified by users
        Prior          % Prior object
        PriorVal       % Parameters of priors  
        Intercept      % Option to add intercept or not (only for testing)
        AutoDiff       % Option to use autodiff (only for testing)
        CutOff         % Cutoff probability for classification making
        Post           % Struct to store training results (maybe not used)        
    end
    
    methods
        % Constructors
        function obj = LogisticRegression(n_features,varargin)
            %LOGISTICREGRESSION Construct an instance of this class
            %   Detailed explanation goes here
            obj.ModelName  = 'LogisticRegression';
            obj.PriorInput = {'Normal',[0,1]};
            obj.Intercept  = true;
            obj.AutoDiff   = false;
            obj.NumParams  = n_features;
            obj.CutOff     = 0.5;

            % Get additional arguments (some arguments are only for testing)
            if nargin > 1
                %Parse additional options
                paramNames = {'AutoDiff'              'Intercept'          'Prior',...
                              'CutOff'};
                paramDflts = {obj.AutoDiff            obj.Intercept        obj.PriorInput,...
                              obj.CutOff};

                [obj.AutoDiff,...
                 obj.Intercept,...
                 obj.PriorInput,...
                 obj.CutOff] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
            end 
            
            % Set prior object using built-in distribution classes
            eval(['obj.Prior=',obj.PriorInput{1}]);
            obj.PriorVal = obj.PriorInput{2};
            
        end
        
        %% Log likelihood
        % Input: 
        %   - data: 2D array. The last column is the responses
        %   - params: Dx1 vector of parameters
        % Output: 
        %   - llh: Log likelihood of the model
        function llh = logLik(obj,data,params)
            
            % Make sure params is a columns
            params = reshape(obj.toOriginalParams(params),obj.NumParams,1);
                        
            % Extract data
            y = data(:,end);
            X = data(:,1:end-1);
            
            % Compute log likelihood
            aux = X*params;
            llh = y.*aux-log(1+exp(aux));
            llh = sum(llh);
            
            
        end

        %% Compute gradient of Log likelihood
        % Input: 
        %   - data: 2D array. The last column is the responses
        %   - params: Dx1 vector of parameters
        % Output: 
        %   - llh_grad: Log likelihood of the model
        function [llh_grad,llh] = logLikGrad(obj,data,params)
            
            % Extract data
            y = data(:,end);
            X = data(:,1:end-1);
            
            % Convert theta (normal) to original distribution
            params = reshape(obj.toOriginalParams(params),obj.NumParams,1);
            
            % Check if auto-diff option is available
            if (obj.AutoDiff)
                % We have to convert params to dlarray to enable autodiff
                params_autodiff = dlarray(params);
                [llh_grad_autodiff,llh_auto_diff] = dlfeval(@obj.logLikGradAutoDiff,data,params_autodiff);
                llh_grad = extractdata(llh_grad_autodiff)';
                llh      = extractdata(llh_auto_diff);
            else
                % Compute gradient of log likelihood
                aux       = X*params;            
                ppi       = 1./(1+exp(-aux));
                llh_grad  = X'*(y-ppi);
                
                % Compute log likelihood
                llh = y.*aux-log(1+exp(aux));
                llh = sum(llh);
            end
        end
 
        %% Compute gradient of Log likelihood using AutoDiff
        % Input: 
        %   - data: 2D array. The last column is the responses
        %   - params: 1xD vector of parameters
        % Output: 
        %   - llh_grad: Log likelihood of the model
        function [llh_grad,llh] = logLikGradAutoDiff(obj,data,params)
                        
            llh = obj.logLik(data,params);
             
            llh_grad = dlgradient(llh,params);
        end
        
        %% Compute log prior of parameters
        % Input: 
        %   - params: the Dx1 vector of parameters
        % Output: 
        %   - llh: Log prior of model parameters       
        function log_prior = logPriors(obj,params)
            
            params = reshape(obj.toOriginalParams(params),obj.NumParams,1);
            
            % Compute log prior
            log_prior = obj.Prior.logPdfFnc(params,obj.PriorVal);
            
        end  
        
        %% Compute gradient of log prior of parameters
        % Input: 
        %   - params: 1xD vector of parameters
        % Output: 
        %   - log_prior_grad: Gradient of log prior of model parameters       
        function [log_prior_grad,log_prior] = logPriorsGrad(obj,params)

            % Compute log prior
            log_prior = obj.Prior.logPdfFnc(params,obj.PriorVal);
            
            % Compute gradient of log prior
            log_prior_grad = obj.Prior.GradlogPdfFnc(params,obj.PriorVal);
        end        
        
        %% Log of Jacobian of all paramters
        % Input: 
        %   - params: the ROW vector of parameters
        % Output: 
        %   - llh: Log prior of model parameters  
        function logjac = logJac(obj,params)
            logjac = 0;
        end
        
        %% Log of Jacobian of all paramters
        % Input: 
        %   - params: the ROW vector of parameters
        % Output: 
        %   - llh: Log prior of model parameters  
        function [logJac_grad,logJac] = logJacGrad(obj,params)
            logJac_grad = 0;
            logJac      = 0;
        end
        
        %% Function to compute h_theta = log lik + log prior
        % Input: 
        %   - data: 2D array. The last column is the responses
        %   - theta: Dx1 vector of parameters
        % Output: 
        %   - h_func: Log likelihood + log prior
        function h_func = hFunction(obj,data,theta)            
            % Transform parameters from normal to original distribution
            params = obj.toOriginalParams(theta);  
            
            % Compute h(theta)
            log_lik = obj.logLik(data,params);
            log_prior = obj.logPriors(params);
            log_jac = obj.logJac(params);
            h_func = log_lik + log_prior + log_jac;
        end
        
        %% Function to compute gradient of h_theta = grad log lik + grad log prior
        % Input: 
        %   - data: 2D array. The last column is the responses
        %   - theta: Dx1 vector of parameters
        % Output: 
        %   - h_func_grad: gradient (Log likelihood + log prior)
        %   - h_func: Log likelihood + log prior
        function [h_func_grad, h_func] = hFunctionGrad(obj,data,theta)
            
            % Transform parameters from normal to original distribution
            params = obj.toOriginalParams(theta);
            
            % Compute h(theta)
            [llh_grad,llh] = obj.logLikGrad(data,params);
            [log_prior_grad,log_prior] = obj.logPriorsGrad(params);
            [logJac_grad,logJac] = obj.logJacGrad(params);
            h_func = llh + log_prior + logJac;
            h_func_grad = llh_grad + log_prior_grad + logJac_grad;
        end        
        
        %% Transform parameters to from normal to original distribution
        function paramsOriginal = toOriginalParams(obj,params)
            paramsOriginal = obj.Prior.toOriginalParams(params);
        end

        %% Transform parameters to from normal to original distribution
        function paramsNormal = toNormalParams(obj,params)
            paramsNormal = obj.Prior.toNormalParams(params);
        end
        
        %% Initialize parameters
        function params = initParams(obj,type,varargin)
            d_theta = obj.NumParams;
            switch type
                case 'MLE' % 2D array of must be provided
                    data = varargin{1};
                    X = data(:,1:end-1);
                    y = data(:,end);
                    params = glmfit(X,y,'binomial','constant','off'); % initialise mu
                case 'Prior'
                    params = obj.Prior.rngFnc(obj.PriorVal,[d_theta,1]);    
                case 'Random' % (only for testing)
                    std_init = varargin{1};
                    params   = normrnd(0,std_init,[d_theta,1]);
                case 'Zeros' % (Only for testing)
                    params = zeros(d_theta,1); 
                otherwise
                    error(['There is no initialization method called ',type,' in the model object!'])
            end
        end
    end
end

