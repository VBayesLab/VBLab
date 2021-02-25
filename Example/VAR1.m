classdef VAR1
    %VAR1 Class to model the VAR(1) model
    
    properties 
        ModelName      % Model name 
        NumParams      % Number of parameters
        Prior          % Prior object
        ParamIdx       % Indexes of model parameters in the vector of variational parameters
        Gamma          % Fix covarian matrix
    end
    
    methods
        % Constructor. This will be automatically called when users create a VAR1 object
        function obj = VAR1(NumSeries)
            % Set value for ModelName and NumParams
            obj.ModelName  = 'VAR1';                 
            obj.NumParams  = NumSeries + NumSeries^2; 
            obj.Prior      = [0,1]; % Use a normal distribution for prior
            obj.ParamIdx.c = 1:NumSeries;
            obj.ParamIdx.A = NumSeries+1:obj.NumParams;
            obj.Gamma      =  0.1*eye(NumSeries);
        end
        
        % Function to compute gradient of h_theta and h_theta
        function [h_func_grad, h_func] = hFunctionGrad(obj,y,theta)
            % Extract size of data
            [m,T] = size(y);
                
            % Extract model properties
            prior_params = obj.Prior;
            d = obj.NumParams;
            idx = obj.ParamIdx;
            gamma = obj.Gamma;
            gamma_inv = gamma^(-1);

            % Extract params from theta
            c = theta(idx.c);                               % c is a column
            A = reshape(theta(idx.A),length(c),length(c));  % A is a matrix
                
            % Log prior
            log_prior = Normal.logPdfFnc(theta,prior_params);
                
            % Log likelihood
            log_llh = 0;
            for t=2:T
                log_llh = log_llh -0.5*(y(:,t) - A*y(:,t-1)-c)' * gamma_inv * (y(:,t) - A*y(:,t-1)-c);
            end  
            log_llh = log_llh - 0.5*m*(T-1)*log(2*pi) - 0.5*(T-1)*log(det(gamma));

            % Compute h_theta
            h_func = log_prior + log_llh;
                
            % Gradient log_prior
            grad_log_prior = Normal.GradlogPdfFnc(theta,prior_params);
                
            % Gradient log_llh;
            grad_llh_c = 0;
            grad_llh_A = 0;
            for t=2:T
                grad_llh_c = grad_llh_c + gamma_inv*(y(:,t) - A*y(:,t-1)-c);
                grad_llh_A = grad_llh_A + kron(y(:,t-1),gamma_inv*(y(:,t) - A*y(:,t-1)-c));
            end
                
            grad_llh = [grad_llh_c;grad_llh_A(:)];
                
            % Compute Gradient of h_theta
            h_func_grad = grad_log_prior + grad_llh;
            
            % Make sure grad_h_theta is a column
            h_func_grad = reshape(h_func_grad,d,1);
        end  
    end
end

