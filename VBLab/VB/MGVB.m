classdef MGVB < VBayesLab
    %MVB Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        GradClipInit       % If doing gradient clipping at the beginning
    end
    
    methods
        function obj = MGVB(mdl,data,varargin)
            %MVB Construct an instance of this class
            %   Detailed explanation goes here
            obj.Method        = 'MGVB';
            obj.GradWeight    = 0.4;    % Small gradient weight is better
            obj.GradClipInit  = 0;      % Sometimes we need to clip the gradient early
            
            % Parse additional options
            if nargin > 2
                paramNames = {'NumSample'             'LearningRate'       'GradWeight'      'GradClipInit' ...      
                              'MaxIter'               'MaxPatience'        'WindowSize'      'Verbose' ...        
                              'InitMethod'            'StdForInit'         'Seed'            'MeanInit' ...       
                              'SigInitScale'          'LBPlot'             'GradientMax' ...
                              'NumParams'             'DataTrain'          'Setting'         'StepAdaptive' ...
                              'SaveParams'};
                paramDflts = {obj.NumSample            obj.LearningRate    obj.GradWeight    obj.GradClipInit ...    
                              obj.MaxIter              obj.MaxPatience     obj.WindowSize    obj.Verbose ...      
                              obj.InitMethod           obj.StdForInit      obj.Seed          obj.MeanInit ...      
                              obj.SigInitScale         obj.LBPlot          obj.GradientMax  ...
                              obj.NumParams            obj.DataTrain       obj.Setting       obj.StepAdaptive ...
                              obj.SaveParams};

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
               obj.HFuntion = mdl;
           end
           
           % Main function to run MGVB
           obj.Post   = obj.fit(data);             
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
            momentum_weight = obj.GradWeight;
            init_scale      = obj.SigInitScale;
            stepsize_adapt  = obj.StepAdaptive;
            max_iter        = obj.MaxIter;
            lb_plot         = obj.LBPlot;
            max_grad        = obj.GradientMax;
            max_grad_init   = obj.GradClipInit;
            hfunc           = obj.HFuntion;
            setting         = obj.Setting;
            verbose         = obj.Verbose;
            save_params     = obj.SaveParams;     

            % Store variational mean in each iteration (if specified)
            if(save_params)
                params_iter = zeros(max_iter,d_theta);
            end  
            
            % Initialization
            iter      = 0;              
            patience  = 0;
            stop      = false; 
            LB_smooth = 0;
            
            % Initialization of mu
            % If initial parameters are not specified, then use some
            % initialization methods
            if isempty(ini_mu)
                mu = normrnd(0,std_init,d_theta,1);
            else % If initial parameters are provided
                mu = ini_mu;
            end
            
            Sig     = init_scale*eye(d_theta); % Initialization of Sig
            c12     = zeros(1,d_theta+d_theta*d_theta);   % Control variate, initilised to be all zero
            Sig_inv = eye(d_theta)/Sig;
            
            gra_log_q_lambda         = zeros(S,d_theta+d_theta*d_theta); % Gradient of log_q
            grad_log_q_h_function    = zeros(S,d_theta+d_theta*d_theta); % (gradient of log_q) x h(theta)
            grad_log_q_h_function_cv = zeros(S,d_theta+d_theta*d_theta);                   % Control_variate version: (gradient of log_q) x (h(theta)-c)
            
            rqmc = utils_normrnd_qmc(S,d_theta);      % Generate standard normal numbers, using quasi-MC
            C_lower = chol(Sig,'lower');
            
            for s = 1:S
                % Parameters in Normal distribution
                theta = mu + C_lower*rqmc(s,:)';
                
                % If handle of function to compute h(theta) is not provided, 
                % then a model object with method of calculating  of h(theta) 
                % must be provided.
                if isempty(hfunc)
                    if (~isempty(obj.Model)) 
                        % Call the hFunction of the model to compute h(theta)
                        h_theta = model.hFunction(data,theta);   
                    else
                        error('An model object of handle of function to compute gradient of h(theta) must be provided!')
                    end                   
                else
                    % If user provide function to directly compute h(theta)
                    % then use it
                    h_theta = hfunc(data,theta,setting);
                end
                
                % Log q_lambda
                log_q_lambda = -d_theta/2*log(2*pi)-1/2*log(det(Sig))-1/2*(theta-mu)'*Sig_inv*(theta-mu);
                
                % h function
                h_function = h_theta - log_q_lambda;

                aux                           = Sig_inv*(theta-mu);
                gra_log_q_mu                  = aux;
                gra_log_q_Sig                 = -1/2*Sig_inv+1/2*aux*(aux');    
                gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*h_function;    
                grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(h_function-c12);    
            end
            
            c12 = zeros(1,d_theta+d_theta*d_theta); 
            for i = 1:d_theta+d_theta*d_theta
                aa     = cov(grad_log_q_h_function(:,i),gra_log_q_lambda(:,i));
                c12(i) = aa(1,2)/aa(2,2);
            end
            Y12 = mean(grad_log_q_h_function_cv)'; % Euclidiance gradient of lower bounf LB
            
            % Gradient clipping at the beginning
            if(max_grad_init>0)
                grad_norm = norm(Y12);
                norm_gradient_threshold = max_grad_init;
                if grad_norm>norm_gradient_threshold
                    Y12 = (norm_gradient_threshold/grad_norm)*Y12;
                end
            end

            % To use manifold GVB for other models, all we need is Euclidiance gradient
            % of LB. All the other stuff below are model-independent.
            gradLB_mu           = Sig*Y12(1:d_theta);                % Natural gradient of LB w.r.t. mu
            gradLB_Sig          = Sig*reshape(Y12(d_theta+1:end),d_theta,d_theta)*Sig; % Natural gradient of LB w.r.t. Sigma
            gradLB_Sig_momentum = gradLB_Sig;                        % Initialise momentum gradient for Sig
            gradLB_mu_momentum  = gradLB_mu;                         % initialise momentum gradient for Sig

            % Prepare for the next iterations
            mu_best   = mu; 
            Sig_best  = Sig; 
            while ~stop   
                
                iter = iter+1;    
                if iter>stepsize_adapt
                    stepsize = eps0*stepsize_adapt/iter;
                else
                    stepsize = eps0;
                end    
                Sig_old = Sig;    
                Sig     = obj.retraction_spd(Sig_old,gradLB_Sig_momentum,stepsize); % retraction to update Sigma
                mu      = mu + stepsize*gradLB_mu_momentum;                       % update mu

                gra_log_q_lambda         = zeros(S,d_theta + d_theta*d_theta); 
                grad_log_q_h_function    = zeros(S,d_theta + d_theta*d_theta); 
                grad_log_q_h_function_cv = zeros(S,d_theta + d_theta*d_theta); % control_variate
                
                lb_log_h = zeros(S,1);
                Sig_inv  = eye(d_theta)/Sig;
                rqmc     = utils_normrnd_qmc(S,d_theta);      
                C_lower  = chol(Sig,'lower');
                for s = 1:S    
                    % Parameters in Normal distribution
                    theta = mu + C_lower*rqmc(s,:)';
                    
                    % If handle of function to compute h(theta) is not provided, 
                    % then a model object with method of calculating  of h(theta) 
                    % must be provided.
                    if isempty(hfunc)
                        if (~isempty(obj.Model)) 
                            % Call the hFunction of the model to compute h(theta)
                            h_theta = model.hFunction(data,theta);   
                        else
                            error('An model object of handle of function to compute gradient of h(theta) must be provided!')
                        end                   
                    else
                        % If user provide function to directly compute h(theta)
                        % then use it
                        h_theta = hfunc(data,theta,setting);
                    end
                    
                    % log q_lambda
                    log_q_lambda = -d_theta/2*log(2*pi)-1/2*log(det(Sig))-1/2*(theta-mu)'*Sig_inv*(theta-mu);
                    
                    h_function = h_theta - log_q_lambda;
                    
                    % To compute the lowerbound
                    lb_log_h(s) = h_function;

                    aux                           = Sig_inv*(theta-mu);
                    gra_log_q_mu                  = aux;
                    gra_log_q_Sig                 = -1/2*Sig_inv+1/2*aux*(aux');    
                    gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                    grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*h_function;    
                    grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(h_function-c12);
                end
                for i = 1:d_theta+d_theta*d_theta
                    aa = cov(grad_log_q_h_function(:,i),gra_log_q_lambda(:,i));
                    c12(i) = aa(1,2)/aa(2,2);
                end  
                Y12 = mean(grad_log_q_h_function_cv)';
                
                % Clipping the gradient
                grad_norm               = norm(Y12);
                norm_gradient_threshold = max_grad;
                if grad_norm > norm_gradient_threshold
                    Y12 = (norm_gradient_threshold/grad_norm)*Y12;
                end

                gradLB_mu  = Sig*Y12(1:d_theta);
                gradLB_Sig = Sig*reshape(Y12(d_theta+1:end),d_theta,d_theta)*Sig;

                zeta = obj.parallel_transport_spd(Sig_old,Sig,gradLB_Sig_momentum); % vector transport to move gradLB_Sig_momentum
                
                % From previous Sig_old to new point Sigma
                gradLB_Sig_momentum = momentum_weight*zeta+(1-momentum_weight)*gradLB_Sig; % update momentum grad for Sigma
                gradLB_mu_momentum  = momentum_weight*gradLB_mu_momentum+(1-momentum_weight)*gradLB_mu; % update momentum grad for mu

                % Lower bound
                LB(iter) = mean(lb_log_h);
                
                % Smooth the lowerbound and store best results 
                if iter>window_size
                    LB_smooth(iter-window_size) = mean(LB(iter-window_size:iter));    % smooth out LB by moving average
                    if LB_smooth(iter-window_size)>=max(LB_smooth)
                        mu_best  = mu; 
                        Sig_best = Sig;
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
                
                % If users want to save variational mean in each iteration
                % Only use when debuging code
                if(save_params)
                    params_iter(iter,:) = mu;
                end
                
            end
 
            % Store output 
            if(save_params)
                Post.muIter = params_iter(1:iter-1,:);
            end
            
            % Store output
            Post.LB_smooth = LB_smooth;
            Post.LB        = LB;
            Post.mu        = mu_best; 
            Post.Sigma     = Sig_best;
            Post.sigma2    = diag(Post.Sigma);
            
            % Plot lowerbound
            if(lb_plot)
                obj.plot_lb(LB_smooth);
            end
        end
        
        %% 
        function zeta = parallel_transport_spd(obj,X, Y, eta)
            E    = sqrtm((Y/X));
            zeta = E*eta*E';
        end
        
        %%
        function Y = retraction_spd(obj,X, eta, t)
            teta      = t*eta;
            symm      = @(X) .5*(X+X');
            Y         = symm(X + teta + .5*teta*(X\teta));
            [~,index] = chol(Y);
            iter      = 1; 
            max_iter  = 5;
            while (index)&&(iter<=max_iter)
                iter      = iter+1;
                t         = t/2;
                teta      = t*eta;
                Y         = symm(X + teta + .5*teta*(X\teta));
                [~,index] = chol(Y);
            end   
            if iter >= max_iter
                Y = X;
            end
        end
    end
end

