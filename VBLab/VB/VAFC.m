classdef VAFC < VBayesLab
    %VAFC Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        NumFactor          % Number of factors
        Adelta             % If ADADELTA is used for optimization
    end
    
    methods
        function obj = VAFC(data,varargin)
            %CGVB Construct an instance of this class
            %   Detailed explanation goes here
            obj.Method       = 'VAFC';
            obj.NumFactor    = 4;
            obj.Adelta.rho   = 0.95;
            obj.Adelta.eps   = 10^-6;
            obj.Optimization = 'Simple';  % Could be 'Adelta'
            obj.SigInitScale = 0.01;
            
            % Parse additional options
            if nargin > 1
                %Parse additional options
                paramNames = {'NumSample'             'LearningRate'       'GradWeight'      ...      
                              'MaxIter'               'MaxPatience'        'WindowSize'      'Verbose' ...        
                              'InitMethod'            'StdForInit'         'Seed'            'MeanInit' ...       
                              'SigInitScale'          'LBPlot'             'GradientMax'     'AutoDiff' ...       
                              'HFuntion'              'GradHFuntion'       'ParamsDim'       'Model' ...
                              'DataTrain'             'Setting'            'StepAdaptive'    'NumFactor',...
                              'SaveParams'            'Optimization'};
                paramDflts = {obj.NumSample            obj.LearningRate    obj.GradWeight    ...    
                              obj.MaxIter              obj.MaxPatience     obj.WindowSize    obj.Verbose ...      
                              obj.InitMethod           obj.StdForInit      obj.Seed          obj.MeanInit ...      
                              obj.SigInitScale         obj.LBPlot          obj.GradientMax   obj.AutoDiff...
                              obj.HFuntion             obj.GradHFuntion    obj.ParamsDim     obj.Model ...
                              obj.DataTrain            obj.Setting         obj.StepAdaptive  obj.NumFactor...
                              obj.SaveParams           obj.Optimization};

                [obj.NumSample,...
                 obj.LearningRate,...
                 obj.GradWeight,...
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
                 obj.NumFactor,...
                 obj.SaveParams,...
                 obj.Optimization] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
           end 
           
           % Set model name if model is specified
           if (~isempty(obj.Model))
               model = obj.Model;
               obj.ModelToFit   = model.ModelName; 
           else
               if(~isempty(obj.DataTrain))
                   data = obj.DataTrain;
               else
                   error('A training data must be specified!')
               end
           end
           
           % Main function to run CGVB
           obj.Post   = obj.fit(data);
        end
        
        %% VB main function
        function Post = fit(obj,data)

            % Extract model object if provided
            if (~isempty(obj.Model))                   % If instance of a model is provided
                model           = obj.Model;
                d_theta         = model.ParamNum;      % Number of parameters
            else                                       %   
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
            momentum_weight = obj.GradWeight;
            num_factor      = obj.NumFactor;
            grad_hfunc      = obj.GradHFuntion;
            setting         = obj.Setting;
            opt             = obj.Optimization;
            
            % Initialization
            iter      = 1;              
            patience  = 0;
            stop      = false; 
            LB_smooth = 0;
            
            % Initialization of mu
            % If initial parameters are not specified, then use some
            % initialization methods
            if isempty(ini_mu)
                switch obj.InitMethod
                    case 'MLE'
                        mu = model.initParams('MLE',data);
                        B = normrnd(0,std_init,d_theta,num_factor);
                        c = init_scale*ones(d_theta,1);
                    case 'Prior'
                        mu = model.initParams('Prior',std_init);
                        B = normrnd(0,std_init,d_theta,num_factor);
                        c = init_scale*ones(d_theta,1);
                    case 'Zeros'
                        mu = zeros(d_theta,1) + std_init;
                        B  = zeros(d_theta,num_factor) + std_init;
                        c  = init_scale*ones(d_theta,1);
                    case 'Random'
                        mu = zeros(d_theta,1) + std_init;
                        B  = normrnd(0,std_init,d_theta,num_factor);
                        c  = init_scale*ones(d_theta,1);
                    otherwise
                        error(['There is no initialization method named ',obj.InitMethod,'!'])
                end
            else % If initial parameters are provided
                mu = ini_mu;
                B = normrnd(0,std_init,d_theta,num_factor);
                c = init_scale*ones(d_theta,1);
            end
            
            %  Column vector variational parameters
            lambda = [mu;B(:);c]; 
            
            if (strcmp(opt,'Adelta'))
                % ADADELTA parameter
                rho            = obj.Adelta.rho;
                eps_step       = obj.Adelta.eps;
                Edelta2_lambda = zeros(length(lambda),1);
                Eg2_lambda     = zeros(length(lambda),1);
            end
            

            % Store all setting to a structure
            param(iter,:) = mu';

            %% First VB iteration
            lb_iter         = zeros(S,1);              
            grad_lb_mu_iter = zeros(S,d_theta);   
            grad_lb_B_iter  = zeros(S,d_theta*num_factor); 
            grad_lb_c_iter  = zeros(S,d_theta);   
            
            % To compute log q_lambda
            Dinv2B = bsxfun(@times,B,1./c.^2);
            Blogdet = log(det(eye(num_factor) + bsxfun(@times,B, 1./(c.^2))'*B)) + sum(log((c.^2)));
    
            rqmc = normrnd(0,1,S,d_theta+num_factor); 
            for s = 1:S
                % Compute model parameters from variational parameters
                U_normal = rqmc(s,:)';
                epsilon1 = U_normal(1:num_factor);
                epsilon2 = U_normal((num_factor+1):end);
                theta    = mu + B*epsilon1 + c.*epsilon2;  % Compute theta
           
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
                
                % Gradient of  log variational distribution
                [L_mu,L_B,L_c] = obj.grad_log_q_function(B,c,epsilon1,epsilon2,grad_h_theta);
                 
                % Gradient of lowerbound
                grad_lb_mu_iter(s,:) = L_mu;
                grad_lb_B_iter(s,:)  = L_B(:);
                grad_lb_c_iter(s,:)  = L_c;
              
                % For lower bound
                Bz_deps      = theta - mu;
                DBz_deps     = bsxfun(@times,Bz_deps,1./c.^2); 
                Half1        = DBz_deps;
                Half2        = Dinv2B/(eye(num_factor) + B'*Dinv2B)*B'*DBz_deps;
                log_q_lambda = - d_theta/2*log(2*pi) - 1/2*Blogdet - 1/2*Bz_deps'*(Half1-Half2);
                lb_iter(s)   = h_theta - log_q_lambda;
            end
            
            % Estimation of lowerbound
            LB(iter) = mean(lb_iter);

            % Gradient of log variational distribution
            grad_lb_mu = mean(grad_lb_mu_iter,1)';
            grad_lb_B  = mean(grad_lb_B_iter,1)';
            grad_lb_c  = mean(grad_lb_c_iter,1)';

            % Natural gradient
            gradient_lambda    = obj.inv_fisher_grad_multifactor(B,c,grad_lb_mu,grad_lb_B,grad_lb_c);
            norm_gradient      = norm(gradient_lambda);
            norm_gradient_seq1 = norm_gradient;
            gradient_bar       = gradient_lambda;

            %% Main VB loop
            while ~stop 
                iter = iter + 1;   
                
                % To compute log q_lambda
                Dinv2B = bsxfun(@times,B,1./c.^2);
                Blogdet = log(det(eye(num_factor) + bsxfun(@times,B, 1./(c.^2))'*B)) + sum(log((c.^2)));
            
                rqmc = normrnd(0,1,S,d_theta+num_factor); 
                for s=1:S
                    % Compute model parameters from variational parameters
                    U_normal = rqmc(s,:)';
                    epsilon1 = U_normal(1:num_factor);
                    epsilon2 = U_normal((num_factor+1):end);
                    theta    = mu + B*epsilon1 + c.*epsilon2; 

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
                
                    % Gradient of  log variational distribution
                    [L_mu,L_B,L_c] = obj.grad_log_q_function(B,c,epsilon1,epsilon2,grad_h_theta);
 
                    % Gradient of lowerbound
                    grad_lb_mu_iter(s,:) = L_mu;
                    grad_lb_B_iter(s,:)  = L_B(:);
                    grad_lb_c_iter(s,:)  = L_c;

                    % For lower bound
                    Bz_deps      = theta - mu;
                    DBz_deps     = bsxfun(@times,Bz_deps,1./c.^2); 
                    Half1        = DBz_deps;
                    Half2        = Dinv2B/(eye(num_factor) + B'*Dinv2B)*B'*DBz_deps;
                    log_q_lambda = - d_theta/2*log(2*pi) - 1/2*Blogdet - 1/2*Bz_deps'*(Half1-Half2);
                    lb_iter(s)   = h_theta - log_q_lambda;
                end
                
                % Estimation of lowerbound
                LB(iter) = mean(lb_iter);

                % Gradient of log variational distribution
                grad_lb_mu = mean(grad_lb_mu_iter,1)';
                grad_lb_B  = mean(grad_lb_B_iter,1)';
                grad_lb_c  = mean(grad_lb_c_iter,1)';

                gradient_lambda          = obj.inv_fisher_grad_multifactor(B,c,grad_lb_mu,grad_lb_B,grad_lb_c);
                grad_norm_current        = norm(gradient_lambda);
                norm_gradient_seq1(iter) = grad_norm_current;
                if norm(gradient_lambda)>max_grad
                    gradient_lambda = (max_grad/norm(gradient_lambda))*gradient_lambda;
                end
                norm_gradient = norm_gradient+norm(gradient_lambda);    
                gradient_bar  = momentum_weight*gradient_bar+(1-momentum_weight)*gradient_lambda;
                
                if (strcmp(opt,'Adelta'))
                    % ADADELTA            
                    grad_lb = mean(grad_lb_iter,1)';
                    Eg2_lambda     = rho*Eg2_lambda + (1-rho)*grad_lb.^2;
                    temp           = sqrt(Edelta2_lambda + eps_step)./sqrt(Eg2_lambda+eps_step);
                    d_lambda       = temp.*grad_lb;
                    lambda         = lambda + d_lambda;
                    Edelta2_lambda = rho*Edelta2_lambda + (1-rho)*d_lambda.^2;
                else
                    if iter>tau_threshold
                        stepsize = eps0*tau_threshold/iter;
                    else
                        stepsize = eps0;
                    end
                    lambda = lambda + stepsize*gradient_bar;
                end

                % Reconstruct variantional parameters
                mu   = lambda(1:d_theta,1);
                vecB = lambda(d_theta+1:d_theta+d_theta*num_factor,1);
                B    = reshape(vecB,d_theta,num_factor);
                c    = lambda(d_theta+d_theta*num_factor+1:end,1);

                % Store parameters in each iteration    
                param(iter,:) = mu';
                
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
            Post.mu        = lambda_best(1:d_theta,1);
            Post.B         = reshape(lambda_best(d_theta+1:d_theta+d_theta*num_factor,1),d_theta,num_factor);
            Post.c         = lambda_best(d_theta+d_theta*num_factor+1:end,1);
            Post.params    = param;
            Post.Sigma     = Post.B*Post.B' + diag(Post.c.^2);
            Post.sigma2    = diag(Post.Sigma);
            
            % If users want to plot the lowerbound
            if(lb_plot)
                obj.plot_lb(LB_smooth);
            end            
        end
        
        %% Gradient of log q_lambda
        function [L_mu,L_B,L_c] = grad_log_q_function(obj,B,c,epsilon1,epsilon2,grad_log_h)
            
            Bz_deps  = B*epsilon1 + c.*eps;  % theta-mu
            Dinv2B   = bsxfun(@times,B,1./c.^2); %D^-2*B
            DBz_deps = bsxfun(@times,Bz_deps,1./c.^2);  %D^-2 * Bz_deps

            Half1 = DBz_deps;
            Half2 = Dinv2B/(eye(obj.NumFactor) + B'*Dinv2B)*B'*DBz_deps;
            L_mu  = grad_log_h + (Half1-Half2);
            L_B   = grad_log_h*epsilon1'+(Half1-Half2)*epsilon1';
            L_c   = grad_log_h.*epsilon2 + (Half1 - Half2).*epsilon2;
            
        end
        
        function prod = inv_fisher_grad_multifactor(obj,B,c,grad1,grad2,grad3)
            %function prod = inverse_fisher_times_grad(b,c,grad)
            % compute the product inverse_fisher x grad
            % B: dxp matrix where p<<d
            [d,p] = size(B);

            % I11 = Siginv = D^(-2) - D^(-2)*B/(eye(p) + B'*D^(-2)*B)*B'*D^(-2);
            Dinv2B = bsxfun(@times,B,1./c.^2); %D^-2*B
            Siginv = diag(1./c.^2) - Dinv2B/(eye(p) + B'*Dinv2B)*Dinv2B';
            I11 = Siginv;

            % I22
            I22 = 2*kron((B'*Siginv*B),Siginv);

            % I33
            Siginv_approx = Siginv.*eye(d);  % Approximate Siginv by zeroing all off-diagonal
            Siginv_approx_D = bsxfun(@times,Siginv_approx',c)'; % Siginv_approx * D
            D_Siginv_approx = bsxfun(@times,Siginv_approx,c);  % D * Siginv_approx
            I33 = 2*D_Siginv_approx .* Siginv_approx_D;

            prod1 = I11\grad1;
            prod2 = I22\grad2;
            prod3 = I33\grad3;
            prod = [prod1;prod2;prod3];
        end
        
        %% Plot lowerbound
        % Call this after running VB 
        function plot_lb(obj,lb)
            plot(lb,'LineWidth',2)
            if(~isempty(obj.Model))
                title(['Lower bound ',obj.Method ,' - ',obj.Model.ModelName])
            else
                title('Lower bound')
            end
            xlabel('Iterations')
            ylabel('Lower bound')
        end
    end
end

