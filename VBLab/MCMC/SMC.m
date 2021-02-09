classdef SMC
    %SMC Likelihood Annealing SMC
    
    properties
        Model           % Instance of model to be fitted
        ModelToFit      % Name of model to be fitted
        SeriesLength    % Length of the series    
        NumParticle     % Number particles
        NumMarkov       % Number of Markov move
        NumAnneal       % Number of Annealing level
        SaveFileName    % Save file name
        SaveAfter       % Save the current results after each 5000 iteration
        ParamsInit      % Initial values of model parameters
        Seed            % Random seed
        Post            % Struct to store posterior samples
        Initialize      % Initialization method
        LogLikelihood   % Handle of the log-likelihood function
        PrintMessage    % Custom message during the sampling phase
        CPU             % Sampling time    
        Verbose         % Turn on of off printed message during sampling phase
        SigScale
        Params
        Marllh          % Marginal likelihood estimate
    end
    
    methods
        function obj = SMC(y,model,varargin)
            %SMC Construct an instance of this class
            %   Detailed explanation goes here
            obj.Model         = model; 
            obj.ModelToFit    = model.ModelName;
            obj.SeriesLength  = length(y);
            obj.NumAnneal     = 10000;
            obj.NumParticle   = 200;
            obj.NumMarkov     = 20;
            obj.Verbose       = 1;
            obj.SigScale      = 0.01;
            
            if nargin > 2
                %Parse additional options
                paramNames = {'NumAnneal'         'NumParticle'      'NumMarkov'...
                              'SigScale'};
                paramDflts = {obj.NumAnneal       obj.NumParticle    obj.NumMarkov...
                              obj.SigScale};

                [obj.NumAnneal,...
                 obj.NumParticle,...
                 obj.NumMarkov,...
                 obj.SigScale] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
            end
            
            % Sampling phase
            obj.Post = obj.fit(y);
        end
        
        function Post = fit(obj,y)
            %FIT Sample a posterior using MCMC
            
            % Extract sampling setting
            model      = obj.Model;
            num_params = model.ParamNum;
            verbose    = obj.Verbose;
            T_anneal   = obj.NumAnneal;     
            M          = obj.NumParticle;           
            K          = obj.NumMarkov;                  
            scale      = obj.SigScale;

            % Initialize particle for the first level using parameters priors
            params = model.initialize(M);

            % Prepare for first annealing stage
            psisq   = ((0:T_anneal)./T_anneal).^3;   % Specify an array of a_p -> each annealing level use 1 a_p 
            ESSall  = zeros(T_anneal,1);             % Store ESS in each level
            log_llh = 0;

            % Calculate log likelihood for all particles in the first level
            llh_calc = zeros(M,1);
            parfor i = 1:M
                params_struct = model.paramToStruct(params(i,:));
                llh_calc(i,1) = model.logLik(y,params_struct);
            end
            
            psisq_current = psisq(1);
            markov_idx = 1;
            t = 2;
            annealing_start = tic;
            
            while t <= T_anneal+1
                disp(['Current annealing level: ',num2str(t)])

                % Reweighting the particles       
                incw = (psisq(t) - psisq_current).*llh_calc;
                max_incw = max(incw);
                w = exp(incw - max_incw);      % Numerical stabability
                W = w./sum(w);                 % Calculate weights for current level
                ESS = 1/sum(W.^2);             % Estimate ESS for particles in the current level

                % Calculate covariance matrix of random walk proposal
                theta = params;
                if markov_idx < 5
                    V = eye(num_params);           
                else
                    est = sum(theta.*(W*ones(1,num_params)));
                    aux = theta - ones(M,1)*est;
                    V = aux'*diag(W)*aux;    
                end

                % If a the current level, the ESS > 80% then skip Markov move -> move
                % to next annealing level
                while ESS >= 0.8*M
                    t = t + 1;
                    % Run until ESS at a certain level < 80%. If reach the last level,
                    % stop and return the particles
                    if (t >= T_anneal+1)
                        t = T_anneal+1;
                        incw = (psisq(t)-psisq_current).*llh_calc;
                        max_incw = max(incw);
                        w = exp(incw-max_incw);
                        W = w./sum(w);
                        ESS = 1/sum(W.^2);
                        break
                    else % If not reach the final level -> keep checking ESS 
                        incw = (psisq(t)-psisq_current).*llh_calc;
                        max_incw = max(incw);
                        w = exp(incw-max_incw);
                        W = w./sum(w);
                        ESS = 1/sum(W.^2); 
                    end
                end

                psisq_current = psisq(t);
                ESSall(t-1) = ESS;
                log_llh = log_llh + log(mean(w)) + max_incw; 

                % Resampling for particles at the current annealing level
                indx     = utils_rs_multinomial(W');
                indx     = indx';
                params   = params(indx,:);
                llh_calc = llh_calc(indx,:);

                % Reset weight
                W = ones(M,1)./M;

                % Running Markov move (MH) for each paticles in the current annealing level
                markov_start = tic;
                accept = zeros(M,1);

                % Prepare for the first markov move 

                log_prior = model.logPriors(params);
                jac       = model.logJac(params);
                post      = log_prior + psisq_current*llh_calc;

                % Markov move K times
                iter = ones(1,M);
                parfor i = 1:M
                    while iter(i) <= K

                        % Transform the original parameters to -> [-Inf , Inf]
                        params_normal = model.toNormalParams(params(i,:));

                        % Using multivariate normal distribution as proposal distribution
                        sample = mvnrnd(params_normal,scale.*V);
                        
                        % Contruct parameter in structs
                        params_star = model.paramToStruct(sample);
                
                        % Calculate log-posterior for proposal samples
                        lik_star       = model.logLik(y,params_star);
                        log_prior_star = model.logPriors(sample);        
                        jac_star       = model.logJac(params_star);
                        post_star      = log_prior_star + psisq_current*lik_star;        

                        % Calculate acceptance probability 
                        r1 = exp(post_star-post(i)+jac(i)-jac_star);
                        C1 = min(1,r1);  

                        % If accept the new proposal sample
                        % Use this uniform random number to accept a proposal sample
                        A1 = rand();
                        if (A1 <= C1)
                            params(i,:) = sample;
                            post(i)     = post_star;
                            jac(i)      = jac_star;
                            llh_calc(i) = lik_star;
                            accept(i)   = accept(i) + 1;
                        end
                        iter(i) = iter(i) + 1;
                    end
                end

                Post.cpu_move(markov_idx) = toc(markov_start);
                Post.accept_store(:,markov_idx) = accept;
                disp(['Markov move time ',num2str(markov_idx),': ',num2str(Post.cpu_move(markov_idx)),'s'])
                markov_idx = markov_idx + 1;
                t = t+1;
            end
            cpu = toc(annealing_start);
            
            % Store output
            Post.params     = params;
            Post.ESS        = ESSall;
            Post.cpu        = cpu;
            Post.log_marllh = log_llh;
        end
    end
end

