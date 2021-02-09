classdef CGVB < VBayesLab
    %CGVB Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        GradWeight1        % Momentum weight 1
        GradWeight2        % Momentum weight 2
    end
    
    methods
        function obj = CGVB(data,varargin)
            %CGVB Construct an instance of this class
            %   Detailed explanation goes here
            obj.Method       = 'CGVB';
            obj.GradWeight1  = 0.9;
            obj.GradWeight2  = 0.9;
            
            % Parse additional options
            if nargin > 1
                paramNames = {'NumSample'             'LearningRate'       'GradWeight1'     'GradWeight2' ...      
                              'MaxIter'               'MaxPatience'        'WindowSize'      'Verbose' ...        
                              'InitMethod'            'StdForInit'         'Seed'            'MeanInit' ...       
                              'SigInitScale'          'LBPlot'             'GradientMax'     'AutoDiff' ...       
                              'HFuntion'              'GradHFuntion'       'ParamsDim'       'Model' ...
                              'DataTrain'             'Setting'            'StepAdaptive'    'SaveParams'};
                paramDflts = {obj.NumSample            obj.LearningRate    obj.GradWeight1   obj.GradWeight2 ...    
                              obj.MaxIter              obj.MaxPatience     obj.WindowSize    obj.Verbose ...      
                              obj.InitMethod           obj.StdForInit      obj.Seed          obj.MeanInit ...      
                              obj.SigInitScale         obj.LBPlot          obj.GradientMax   obj.AutoDiff...
                              obj.HFuntion             obj.GradHFuntion    obj.ParamsDim     obj.Model ...
                              obj.DataTrain            obj.Setting         obj.StepAdaptive  obj.SaveParams};

                [obj.NumSample,...
                 obj.LearningRate,...
                 obj.GradWeight1,...
                 obj.GradWeight2,...
                 obj.MaxIter,...
                 obj.MaxPatience,...
                 obj.WindowSize,...
                 obj.Verbose,...
                 obj.InitMethod,...
                 obj.StdForInit,...
                 obj.Seed,...
                 obj.MeanInit,...
                 obj.SigInitScale,...
                 obj.LBPlot,...
                 obj.GradientMax,...
                 obj.AutoDiff,...
                 obj.HFuntion,...
                 obj.GradHFuntion,...
                 obj.ParamsDim,...
                 obj.Model,...
                 obj.DataTrain,...
                 obj.Setting,...
                 obj.StepAdaptive,...
                 obj.SaveParams] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
           end 
           
           % Set model name if model is specified
           if (~isempty(obj.Model))
               model = obj.Model;
               obj.ModelToFit = model.ModelName; 
           end
           
           % Main function to run CGVB
           obj.Post   = obj.fit(data);  
        end
        
        %% VB main function 
        function Post = fit(obj,data)
            
            % Extract model object if provided 
            if (~isempty(obj.Model))                  
                model           = obj.Model;
                d_theta         = model.ParamNum;      % Number of parameters
            else  % If model object is not provided, number of parameters must be provided  
                if (~isempty(obj.ParamsDim))
                    d_theta = obj.ParamsDim;
                else
                    error('Number of model parameters have to be specified!')
                end
            end
            
            % Unload training parameters (only for convenience)
            std_init        = obj.StdForInit;
            eps0            = obj.LearningRate;
            S               = obj.NumSample;
            ini_mu          = obj.MeanInit;
            window_size     = obj.WindowSize;
            max_patience    = obj.MaxPatience;
            init_scale      = obj.SigInitScale;
            tau_threshold   = obj.StepAdaptive;
            max_iter        = obj.MaxIter;
            lb_plot         = obj.LBPlot;
            max_grad        = obj.GradientMax;
            momentum_beta1  = obj.GradWeight1;
            momentum_beta2  = obj.GradWeight2;
            grad_hfunc      = obj.GradHFuntion;
            setting         = obj.Setting;
            
            % Initialization
            iter      = 1;              
            patience  = 0;
            stop      = false; 
            LB_smooth = 0;
            
            % Number of variational parameters
            d_lambda = d_theta + d_theta*(d_theta+1)/2;

            % Initialization of mu
            % If initial parameters are not specified, then use some
            % initialization methods
            if isempty(ini_mu)
                switch obj.InitMethod
                    case 'MLE'
                        mu = model.initParams('MLE',data);
                    case 'Prior'
                        mu = model.initParams('Random',std_init);
                    case 'Zeros'  % No need, only for testing
                        mu = zeros(d_theta,1);
                    case 'Random'
                        mu = normrnd(0,std_init,d_theta,1);
                    otherwise
                        error(['There is no initialization method named ',obj.InitMethod,'!'])
                end
            else % If initial parameters are provided
                mu = ini_mu;
            end
            
            % Initialize variational parameters
            L      = init_scale*eye(d_theta);
            lambda = [mu;vech(L)];

            % Pre-allocation
            grad_LB  = zeros(S,d_lambda);
            h_lambda = zeros(S,1);
            rqmc     = normrnd(0,1,S,d_theta); 
  
            for s = 1:S  
                % Parameters in Normal distribution
                varepsilon = rqmc(s,:)';
                theta      = mu+L*varepsilon;  % Theta -> Dx1 column

                % Gradient of q_lambda. This function is independent to the
                % model 
                [grad_log_q,log_q] = obj.log_q_grad(theta,mu,L);
                
                % If handle of function to compute gradient of h(theta),
                % then a model object with method of calculating gradient
                % of h(theta) must be provided.
                if isempty(grad_hfunc)
                    if (~isempty(obj.Model)) 
                        % Call the hFunctionGrad of the model to compute
                        % h(theta) and gradient of h(theta)
                        [grad_h_theta,h_theta] = model.hFunctionGrad(data,theta);   
                    else
                        error('An model object of handle of function to compute gradient of h(theta) must be provided!')
                    end                   
                else
                    % If user provide function to directly compute gradient
                    % h theta then use it
                    [grad_h_theta,h_theta] = grad_hfunc(data,theta,setting);
                end
                
                % Make sure gradient is a column
                grad_h_theta = reshape(grad_h_theta,length(grad_h_theta),1);
                
                % Compute h_lambda and gradient of h_lambda
                h_lambda(s) = h_theta - log_q;
                grad_h_lambda = grad_h_theta - grad_log_q ;
                    
                % Gradient of lowerbound
                grad_LB(s,:)  = [grad_h_lambda;utils_vech(grad_h_lambda*(varepsilon'))]';
                
            end
            grad_LB = mean(grad_LB)';
            LB      = mean(h_lambda);

            % Gradient clipping to avoid exploded gradient
            grad_norm = norm(grad_LB);
            if norm(grad_LB) > max_grad
                grad_LB = (max_grad/grad_norm)*grad_LB;
            end

            g_adaptive     = grad_LB; 
            v_adaptive     = g_adaptive.^2; 
            g_bar_adaptive = g_adaptive; 
            v_bar_adaptive = v_adaptive; 
            
            % Run main VB iterations 
            while ~stop    
                iter     = iter+1;
                mu       = lambda(1:d_theta);
                L        = utils_vechinv(lambda(d_theta+1:end),2);

                grad_LB  = zeros(S,d_lambda);
                h_lambda = zeros(S,1);
                rqmc     = normrnd(0,1,S,d_theta); 
                for s = 1:S    
                    % Parameters in Normal distribution
                    varepsilon = rqmc(s,:)';
                    theta      = mu+L*varepsilon;

                    % Gradient of q_lambda. This function is independent to the
                    % model 
                    [grad_log_q,log_q] = obj.log_q_grad(theta,mu,L);

                    % If handle of function to compute gradient of h(theta),
                    % then a model object with method of calculating gradient
                    % of h(theta) must be provided.
                    if isempty(grad_hfunc)
                        if (~isempty(obj.Model)) 
                            % Call the hFunctionGrad of the model to compute
                            % h(theta) and gradient of h(theta)
                            [grad_h_theta,h_theta] = model.hFunctionGrad(data,theta);   
                        else
                            error('An model object of handle of function to compute gradient of h(theta) must be provided!')
                        end                   
                    else
                        % If user provide function to directly compute gradient
                        % h theta then use it
                        [grad_h_theta,h_theta] = grad_hfunc(data,theta,setting);
                    end

                    % Make sure gradient is a column
                    grad_h_theta = reshape(grad_h_theta,length(grad_h_theta),1);

                    % Compute h_lambda and gradient of h_lambda
                    h_lambda(s) = h_theta - log_q;
                    grad_h_lambda = grad_h_theta - grad_log_q ;

                    % Gradient of lowerbound
                    grad_LB(s,:)  = [grad_h_lambda;utils_vech(grad_h_lambda*(varepsilon'))]';
                end
                
                grad_LB = mean(grad_LB)';

                % gradient clipping
                grad_norm = norm(grad_LB);    
                if norm(grad_LB)>max_grad
                    grad_LB = (max_grad/grad_norm)*grad_LB;
                end

                g_adaptive     = grad_LB; 
                v_adaptive     = g_adaptive.^2; 
                g_bar_adaptive = momentum_beta1*g_bar_adaptive+(1-momentum_beta1)*g_adaptive;
                v_bar_adaptive = momentum_beta2*v_bar_adaptive+(1-momentum_beta2)*v_adaptive;

                % After a specified number of iterations, make the step
                % size smaller. This can be modified to implement more
                % sotiphicated adaptive learning rate methods.
                if iter>=tau_threshold
                    stepsize = eps0*tau_threshold/iter;
                else
                    stepsize = eps0;
                end
                
                % Update new lambda
                lambda = lambda + stepsize*g_bar_adaptive./sqrt(v_bar_adaptive);
                
                % Estimate the lowerbound at the current iteration
                LB(iter) = mean(h_lambda);

                % Smooth the lowerbound
                if iter>=window_size
                    LB_smooth(iter-window_size+1) = mean(LB(iter-window_size+1:iter));
                end

                % Check for early stopping
                if (iter>window_size)&&(LB_smooth(iter-window_size+1)>=max(LB_smooth))
                    lambda_best = lambda;
                    patience = 0;
                else
                    patience = patience+1;
                end

                if (patience>max_patience)||(iter>max_iter) 
                    stop = true; 
                end 

                % Display training information
                if(obj.Verbose)
                    if iter> window_size
                        disp(['Iter: ',num2str(iter),'| LB: ',num2str(LB_smooth(iter-window_size))])
                    else
                        disp(['Iter: ',num2str(iter),'| LB: ',num2str(LB(iter))])
                    end
                end
                
            end
            
            % Store output 
            Post.LB_smooth = LB_smooth;
            Post.LB        = LB;
            Post.lambda    = lambda_best;
            Post.mu        = lambda(1:d_theta);
            Post.L         = utils_vechinv(lambda(d_theta+1:end),2);
            Post.Sigma     = L*(L');
            Post.sigma2    = diag(Post.Sigma);
            
            % If users want to plot the lowerbound
            if(lb_plot)
                obj.plot_lb(LB_smooth);
            end
            
        end
        
        %% Gradient of log_q_lambda. This is independent to the model
        % Log pdf of multivariate normal distribution
        function [grad_log_q,log_q] = log_q_grad(obj,theta,mu,L)
            d          = length(theta);
            Sigma      = L*(L');
            log_q      = -d/2*log(2*pi)-1/2*log(det(Sigma))-1/2*(theta-mu)'*(Sigma\(theta-mu));
            grad_log_q = -Sigma\(theta-mu);
            
        end        
    end
end

