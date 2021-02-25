classdef NAGVAC < VBayesLab
    %NAGVAC Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        GradClipInit       % If doing gradient clipping at the beginning
    end
    
    methods
        function obj = NAGVAC(mdl,data,varargin)
            %NAGVAC Construct an instance of this class
            %   Detailed explanation goes here
            obj.Method       = 'NAGVAC';
            obj.WindowSize   = 30;
            obj.NumSample    = 10;
            obj.LearningRate = 0.01;
            obj.MaxIter      = 5000;
            obj.MaxPatience  = 20;
            obj.StdForInit   = 0.01;
            obj.StepAdaptive = obj.MaxIter/2;
            obj.GradWeight   = 0.9;
            obj.LBPlot       = true;
            obj.GradientMax  = 100; 
            obj.InitMethod   = 'Random';
            obj.Verbose      = true;
            obj.SaveParams   = false;
 
            % Parse additional options
            if nargin > 2
                paramNames = {'NumSample'             'LearningRate'       'GradWeight'      'GradClipInit'...      
                              'MaxIter'               'MaxPatience'        'WindowSize'      'Verbose' ...        
                              'InitMethod'            'StdForInit'         'Seed'            'MeanInit' ...       
                              'SigInitScale'          'LBPlot'             'GradientMax'     'AutoDiff' ...       
                              'HFuntion'              'NumParams'          'DataTrain'       'Setting'...
                              'StepAdaptive'          'SaveParams'};
                paramDflts = {obj.NumSample            obj.LearningRate    obj.GradWeight    obj.GradClipInit ...    
                              obj.MaxIter              obj.MaxPatience     obj.WindowSize    obj.Verbose ...      
                              obj.InitMethod           obj.StdForInit      obj.Seed          obj.MeanInit ...      
                              obj.SigInitScale         obj.LBPlot          obj.GradientMax   obj.AutoDiff ...
                              obj.HFuntion             obj.NumParams       obj.DataTrain     obj.Setting ...
                              obj.StepAdaptive         obj.SaveParams};

                [obj.NumSample,...
                 obj.LearningRate,...
                 obj.GradWeight,...
                 obj.GradClipInit,...
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
                 obj.NumParams,...
                 obj.DataTrain,...
                 obj.Setting,...
                 obj.StepAdaptive,...
                 obj.SaveParams] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
           end        
           
           % Check if model object or function handle is provided
           if (isobject(mdl)) % If model object is provided
               obj.Model = mdl;
               obj.ModelToFit = obj.Model.ModelName; % Set model name if model is specified
           else % If function handle is provided
               obj.GradHFuntion = mdl;
           end
           
           % Main function to run NAGVAC
           obj.Post = obj.fit(data);  
            
        end
        
        %% VB main function
        function Post = fit(obj,data)

            % Extract model object if provided 
            if (~isempty(obj.Model))                  
                model           = obj.Model;
                d_theta         = model.NumParams;      % Number of parameters
            else  % If model object is not provided, number of parameters must be provided  
                if (~isempty(obj.NumParams))
                    d_theta = obj.NumParams;
                else
                    error('Number of model parameters have to be specified!')
                end
            end
            
            % Extract sampling setting
            std_init        = obj.StdForInit;
            eps0            = obj.LearningRate;
            S               = obj.NumSample;
            ini_mu          = obj.MeanInit;
            window_size     = obj.WindowSize;
            max_patience    = obj.MaxPatience;
            init_scale      = obj.SigInitScale;
            momentum_weight = obj.GradWeight;
            tau_threshold   = obj.StepAdaptive;
            max_iter        = obj.MaxIter;
            lb_plot         = obj.LBPlot;
            max_grad        = obj.GradientMax;
            grad_hfunc      = obj.GradHFuntion;
            setting         = obj.Setting;
            verbose         = obj.Verbose;
            save_params     = obj.SaveParams;
 
            % Store variational mean in each iteration (if specified)
            if(save_params)
                params_iter = zeros(max_iter,d_theta);
            end  
            
            % Initialization
            iter        = 1;              
            patience    = 0;
            stop        = false; 
            LB_smooth   = 0;
            lambda_best = [];
            
            % Initialization of mu
            % If initial parameters are not specified, then use some
            % initialization methods
            if isempty(ini_mu)
                mu = normrnd(0,std_init,d_theta,1);
            else % If initial parameters are provided
                mu = ini_mu;
            end
            
            b = normrnd(0,std_init,d_theta,1);     
            c = init_scale*ones(d_theta,1);
            
            lambda             = [mu;b;c];            % Variational parameters vector
            lambda_seq(iter,:) = lambda';

            % Store all setting to a structure
            param(iter,:) = mu';

            %% First VB iteration
            rqmc          = normrnd(0,1,S,d_theta+1); 
            grad_lb_iter  = zeros(S,3*d_theta);       % Store gradient of lb over S MC simulations
            lb_first_term = zeros(S,1);               % To estimate the first term in lb = E_q(log f)-E_q(log q)

            for s = 1:S
                % Parameters in Normal distribution
                U_normal = rqmc(s,:)';
                epsilon1 = U_normal(1);
                epsilon2 = U_normal(2:end);
                theta    = mu + b*epsilon1 + c.*epsilon2;  % Compute Theta
                
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
                grad_log_q = obj.grad_log_q_function(b,c,theta,mu);

                % Gradient of h(theta) and lowerbound
                grad_theta        = grad_h_theta - grad_log_q;
                grad_lb_iter(s,:) = [grad_theta;epsilon1*grad_theta;epsilon2.*grad_theta]';

                % for lower bound
                lb_first_term(s) = h_theta;
                
            end
            
            % Estimation of lowerbound
            logdet   = log(det(1 + (b./(c.^2))'*b)) + sum(log((c.^2)));
            lb_log_q = -0.5*d_theta*log(2*pi) - 0.5*logdet - d_theta/2; % Mean of log-q -> mean(log q(theta))
            LB(iter) = mean(lb_first_term) - lb_log_q;

            % Gradient of log variational distribution
            grad_lb         = (mean(grad_lb_iter))';
            gradient_lambda = obj.inverse_fisher_times_grad(b,c,grad_lb);
            gradient_bar    = gradient_lambda;

            %% Main VB loop
            while ~stop
                
                % If users want to save variational mean in each iteration
                % Only use when debuging code
                if(save_params)
                    params_iter(iter,:) = mu;
                end
                
                iter = iter + 1;
                rqmc = normrnd(0,1,S,d_theta+1); 
                grad_lb_iter  = zeros(S,3*d_theta); % store gradient of lb over S MC simulations
                lb_first_term = zeros(S,1); % to estimate the first term in lb = E_q(log f)-E_q(log q)
                for s=1:S
                    % Parameters in Normal distribution
                    U_normal = rqmc(s,:)';
                    epsilon1 = U_normal(1);
                    epsilon2 = U_normal(2:end);
                    theta    = mu + b*epsilon1 + c.*epsilon2;

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
                    grad_log_q = obj.grad_log_q_function(b,c,theta,mu);

                    % Gradient of h(theta) and lowerbound
                    grad_theta        = grad_h_theta - grad_log_q;
                    grad_lb_iter(s,:) = [grad_theta;epsilon1*grad_theta;epsilon2.*grad_theta]';

                    % for lower bound
                    lb_first_term(s) = h_theta;
                end

                % Estimation of lowerbound
                logdet   = log(det(1 + (b./(c.^2))'*b)) + sum(log((c.^2)));
                lb_log_q = -0.5*d_theta*log(2*pi) - 0.5*logdet - d_theta/2; % Mean of log-q -> mean(log q(theta))
                LB(iter) = mean(lb_first_term) - lb_log_q;

                % Gradient of log variational distribution
                grad_lb         = (mean(grad_lb_iter))';
                gradient_lambda = obj.inverse_fisher_times_grad(b,c,grad_lb);

                % Gradient clipping
                grad_norm = norm(gradient_lambda);
                norm_gradient_threshold = max_grad;
                if norm(gradient_lambda) > norm_gradient_threshold
                    gradient_lambda = (norm_gradient_threshold/grad_norm)*gradient_lambda;
                end

                gradient_bar = momentum_weight*gradient_bar + (1-momentum_weight)*gradient_lambda;     

                if iter > tau_threshold
                    stepsize = eps0*tau_threshold/iter;
                else
                    stepsize = eps0;
                end
                lambda = lambda + stepsize*gradient_bar;
                lambda_seq(iter,:) = lambda';

                % Reconstruct variantional parameters
                mu = lambda(1:d_theta,1);
                b  = lambda(d_theta+1:2*d_theta,1);
                c  = lambda(2*d_theta+1:end);

                % Store parameters in each iteration    
                param(iter,:) = mu';

                if iter > window_size  
                    LB_smooth(iter-window_size) = mean(LB(iter-window_size+1:iter));
                    if LB_smooth(end)>= max(LB_smooth)
                        lambda_best = lambda;
                        patience = 0;
                    else
                        patience = patience + 1;
                    end   
                end
                if (patience>max_patience)||(iter>max_iter) 
                    stop = true;
                end

                % Display training information
                if(verbose)
                    if iter> window_size
                        disp(['Iter: ',num2str(iter),'| LB: ',num2str(LB_smooth(iter-window_size))])
                    else
                        disp(['Iter: ',num2str(iter),'| LB: ',num2str(LB(iter))])
                    end
                end
               
            end
            
            % Store output 
            if(save_params)
                Post.muIter = params_iter(1:iter-1,:);
            end

            % If the algorithm stops too early
            if(isempty(lambda_best))
                lambda_best = lambda;
            end
            
            % Store final results
            Post.LB_smooth = LB_smooth;
            Post.LB        = LB; 
            Post.lambda    = lambda_best;
            Post.mu        = lambda_best(1:d_theta); 
            Post.b         = lambda_best(d_theta+1:2*d_theta);
            Post.c         = lambda_best(2*d_theta+1:end);
            Post.Sigma     = Post.b*Post.b' + diag(Post.c.^2);
            Post.sigma2    = diag(Post.Sigma);
            
            % Plot lowerbound
            if(lb_plot)
                obj.plot_lb(LB_smooth);
            end            
            
        end
        
        %% Obtain samples from the estimate VB
        %  index: Indexes of parameter 
        function Sample = sampleFromVB(obj,Post,n)            
            mu    = Post.mu;
            b     = Post.b;
            c     = Post.c;
            Sigma = b*b'+ diag(c.^2);
            
            Sample = mvnrnd(mu,Sigma,n);
        end      
        
        %% I^-1 x grad
        function prod = inverse_fisher_times_grad(obj,b,c,grad)
            d     = length(b);
            grad1 = grad(1:d);
            grad2 = grad(d+1:2*d);
            grad3 = grad(2*d+1:end);

            c2 = c.^2;
            b2 = b.^2;

            prod1 = (b'*grad1)*b+(grad1.*c2);

            alpha     = 1/(1+sum(b2./c2));
            Cminus    = diag(1./c2);
            Cminus_b  = b./c2;
            Sigma_inv = Cminus-alpha*(Cminus_b*Cminus_b');

            A11_inv = (1/(1-alpha))*((1-1/(sum(b2)+1-alpha))*(b*b')+diag(c2));

            C   = diag(c);
            A12 = 2*(C*Sigma_inv*b*ones(1,d)).*Sigma_inv;
            A21 = A12';
            A22 = 2*C*(Sigma_inv.*Sigma_inv)*C;

            D     = A22-A21*A11_inv*A12;
            prod2 = A11_inv*grad2+(A11_inv*A12)*(D\A21)*(A11_inv*grad2)-(A11_inv*A12)*(D\grad3);
            prod3 = -(D\A21)*(A11_inv*grad2)+D\grad3;

            prod  = [prod1;prod2;prod3];            
        end
        
        %% Gradient of log q_lambda
        function grad_log_q = grad_log_q_function(obj,b,c,theta,mu)
            x          = theta - mu;
            d          = b./c.^2;
            grad_log_q = -x./c.^2+(d'*x)/(1+(d'*b))*d;
        end
        
    end
end

