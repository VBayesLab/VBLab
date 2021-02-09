classdef GARCH < handle & matlab.mixin.CustomDisplay
    %GARCH Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        ModelName              % Model name 
        Distribution           % Distribution of the innovation 
        P                      % Number of lags of the historial Data
        Q                      % Number of lags of the historial volatility
        OffSet                 % Flag to allow the constant term in the volatility equation
        Params                 % Struct to store model parameters
        ParamsNum              % Number of parameters
        ParamsName             % Parameter names
        ParamsDim              % Dimensionality of model parameters
        ParamValues            % Current values of all parameters
        ParamsParticle         % Number of parameter particles
        Post                   % Struct to store posterior samples and information about estimation
        Forecast               % Struct to store forecast results
        Diagnosis              % Struct to store in-sample diagnotis
        VolInit                % The initialization for volatility
        ConditionalVariance    % Handle of function to define the state transition equation
        Measurement            % Handle of function to define the measurement equation
        LogLikelihood          % Handle of the log-likelihood function
        LogPriors              % Log of priors
        StateInitialization    % Start value for state
        Initialization         % 'Prior' or an array of initial values
    end

    %% Method for object display
    methods (Access = protected)
       % Display object properties
       function propgrp = getPropertyGroups(~)
          proplist = {'Distribution','ParamsNum'};
          propgrp = matlab.mixin.util.PropertyGroup(proplist);
       end
       
       % Display footer
       function header = getHeader(obj)
          if ~isscalar(obj)
             header = getHeader@matlab.mixin.CustomDisplay(obj);
          else
             newHeader1 = ['    GARCH(',num2str(obj.P),',',num2str(obj.Q),') model:'];
             newHeader2 =  '    ----------------------------';
             newHeader = {newHeader1
                          newHeader2};
             header = sprintf('%s\n',newHeader{:});
          end
      end
    end
    
    
    %% Class Constructor
    methods
        function obj = GARCH(varargin)
            
            % GARCH(1,1) by default
            obj.P = 1; 
            obj.Q = 1;
            obj.OffSet  = true;
            obj.VolInit = 0.1;  
            
            % Some default values
            obj.ParamsParticle = 1;
            
            % Set up display information
            obj.ModelName      = ['GARCH(',num2str(obj.P),',',num2str(obj.Q),')'];
            obj.Distribution   = 'Gaussian';
            
            % Define default values for model parameters
            obj.Params        = [];     % Clear Params created from superclass's constructor
            
            obj.Params.omega  = Parameter('Name','omega',...
                                          'PriorDist','Uniform',...
                                          'PriorParams',[0,10],...
                                          'Transform','Log');
            obj.Params.alpha  = Parameter('Name','alpha',...
                                          'PriorDist','Uniform',...
                                          'PriorParams',[0,1]);
            obj.Params.beta   = Parameter('Name','beta',...
                                          'PriorDist','Uniform',...
                                          'PriorParams',[0,1]);

            obj.Initialization      = 'Prior';
            
            % Set other properies based on pre-defined parameters
            obj = obj.setPropertiesFnc();
            
            % If user specifies custom state transision and measurement
            % equations
            if nargin > 0
                paramNames = {'Measurement'      'ConditionalVariance'     'LogLikelihood' ...
                              'VolInit'};
                          
                paramDflts = {obj.Measurement    obj.ConditionalVariance   obj.LogLikelihood ...
                              obj.VolInit};

               [obj.Measurement,...
                obj.ConditionalVariance,...
                obj.LogLikelihood,...
                obj.VolInit] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                  
            end
        end
    end
    
    %% Class methods
    
    methods
        %% Set class properties based on users' input
        function obj = setPropertiesFnc(obj,varargin)
            obj.ParamsName     = fieldnames(obj.Params);
            obj.ParamsNum      = length(obj.ParamsName);
        end
        
        %% Get a sample of parameters from prior
        function Theta = randomFromPriorFnc(obj,varargin)
            params      = obj.Params;
            num_params  = obj.ParamsNum;
            params_name = obj.ParamsName;
            if(isempty(varargin))
                dim = [1,1];
            else
                dim = varargin{1};
            end
            
            % Need to check the stationarity constraints for every pair of
            % alpha and beta
            for idx=1:max(dim)
                flag = 0;
                while(~flag)
                    for i=1:num_params
                        param_i      = params.(params_name{i});
                        prior        = param_i.Prior;
                        Theta.(params_name{i})(idx,1) = prior.randomGeneratorFnc([1,1]);
                        ThetaTemp.(params_name{i}) = Theta.(params_name{i})(idx,1);
                    end
                    flag = obj.checkConstraintFnc(ThetaTemp);
                end
            end
        end

        %% Calculate log priors and log jacobian
        function [log_prior,log_jac] = logPriorFnc(obj,theta,varargin)
            params      = obj.Params;
            num_params  = obj.ParamsNum;
            params_name = obj.ParamsName;
            
%             M         = obj.ParamsParticle;
%             log_prior = zeros(M,1);
%             log_jac   = zeros(M,1);

            log_jac   = 0;
            log_prior = 0;
            
            % Loop through all params
            for i=1:num_params
                param_i = params.(params_name{i});
                prior_i = param_i.Prior;
                theta_i = theta.(params_name{i});
                
                % For log-jacobian
                log_jac = log_jac + param_i.logJacobianFnc(theta_i);
                
                % For log-prior
                log_prior = log_prior + prior_i.logPdfFnc(theta_i);
            end            
        end
        
        %% Calculate log likelihood
        function log_llh = logLikelihoodFnc(obj,Theta,Obs,varargin)
            
            % Pre-allocation
            M      = length(Theta.alpha);
            T      = length(Obs);
            sigma2 = zeros(M,T);
            
            if(~obj.OffSet)
                obj.ParamValues.omega = 0;
            end

            sigma2(:,1) = obj.VolInit;
            
            for i = 2:T
                sigma2(:,i) = Theta.omega + Theta.alpha*Obs(i-1)^2 + Theta.beta.*sigma2(:,i-1);
%                 sigma2(:,i) = obj.conditionalVarianceFnc(Theta,Obs(i-1),sigma2(:,i-1));        
            end
%             log_llh = sum(obj.measurementFnc(Theta,sigma2,Obs),2); 
            log_llh = sum(-0.5*log(2*pi) - 0.5*log(sigma2) - 0.5*Obs.^2./sigma2,2);
        end
    
        %% Convert parameters to an [-Inf,+Inf] array for random walk proposal
        function ThetaOut = transformFnc(obj,ThetaIn)
            % Input : Struct
            % Output: Array
            M           = obj.ParamsParticle;
            params      = obj.Params;
            num_params  = obj.ParamsNum;
            params_name = obj.ParamsName;
            ThetaOut    = zeros(M,num_params);
            for i=1:num_params
                param_i  = params.(params_name{i});
                ThetaOut(:,i) = param_i.transformFnc(ThetaIn.(params_name{i}));
            end
        end 
        
    
        %% Convert parameters back to original scale after random walk proposal
        function ThetaOut = invTransformFnc(obj,ThetaIn)
            % Input : Array
            % Output: Struct
            params      = obj.Params;
            num_params  = obj.ParamsNum;
            params_name = obj.ParamsName;
            for i=1:num_params
                param_i  = params.(params_name{i});
                ThetaOut.(params_name{i}) = param_i.invTransformFnc(ThetaIn(i));
            end
        end
    
        %% Assign model parameters from a matrix 
        % Each column of the matrix is the samples of a parameter
        function ThetaStruct = arrayToStruct(obj,ThetaArray)
            num_params  = obj.ParamsNum;
            params_name = obj.ParamsName;     
            for i=1:num_params
                ThetaStruct.(params_name{i}) = ThetaArray(:,i);
            end
        end
        
        %% Resampling parameters with new indexes
        function ParamsOut = resampling(obj,ParamsIn,Index) 
            for i=1:obj.ParamsNum
                params_i = ParamsIn.(obj.ParamsName{i});
                ParamsOut.(obj.ParamsName{i}) = params_i(Index);
            end
        end
        
        %% Simulate a GARCH series
        % Input should be a struct
        function [Y,V] = simulate(obj,Theta,NumPaths,NumSamples,V0,Y0)
            
            V = zeros(NumPaths,NumSamples);
            Y = zeros(NumPaths,NumSamples);
            
            % Initialization
            V(:,1) = V0;
            Y(:,1) = Y0;
            
            for i=2:NumSamples
                V(:,i) = obj.conditionalVarianceFnc(Theta,Y(:,i-1),V(:,i-1));
                Y(:,i) = normrnd(0,sqrt(V(:,i)));
            end
        end
        
        %% Forecast with a GARCH model

        %% Change only parameter's member at a specified index        
        function ThetaIn = updateParamsAtIndexFnc(obj,ThetaIn,Theta,Index)
            params_name = obj.ParamsName; 
            for i=1:obj.ParamsNum
                ThetaIn.(params_name{i})(Index) = Theta.(params_name{i});
            end
        end
        
        %% Constraint to be stationary
        function isAccept = checkConstraintFnc(obj,Theta)
            isAccept = 0; 
            if (Theta.alpha + Theta.beta < 1)
                isAccept = 1;
            end
        end
        %% Define measurement function for GARCH model
        function logYgivenSigma = measurementFnc(obj,Theta,CurrentVol,CurrentObs)
            logYgivenSigma = -0.5*log(2*pi) - 0.5*log(CurrentVol) - 0.5*CurrentObs.^2./CurrentVol;
        end
        
        %% Define state transition function for GARCH model
        function NewVol = conditionalVarianceFnc(obj,Theta,OldObs,OldVol,varagin)
            NewVol = Theta.omega + Theta.alpha*OldObs.^2 + Theta.beta.*OldVol;
        end        
    end
end
