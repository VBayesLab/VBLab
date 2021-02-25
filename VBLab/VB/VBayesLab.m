classdef VBayesLab
    %MODEL Abstract classes for models
    
    properties 
        Method             % VB method -> a name
        Model              % Instance of the model to sample from
        ModelToFit         % Name of model to be fitted
        NumSample          % Number of samples to estimate the likelihood and gradient of likelihood
        GradWeight         % Momentum weight
        LearningRate       % Learning rate decay factor 
        MaxIter            % Maximum number of VB iterations
        MaxPatience        % Maximum number of patiences for early stopping
        WindowSize         % Smoothing window
        ParamsInit         % Initial values of model parameters
        NumParams          % Number of model parameters
        Seed               % Random seed
        Post               % Struct to store estimation results
        Verbose            % Turn on of off printed message during sampling phase
        StdForInit         % Std of the normal distribution to initialize VB params
        MeanInit           % Pre-specified values of mean(theta)
        SigInitScale       % A constant to scale up or down std of normal distribution
        StepAdaptive       % From this iteration, stepsize is reduced 
        LBPlot             % If user wants to plot the lowerbound at the end
        GradientMax        % For gradient clipping
        InitMethod         % Method to initialize mu (variational mean)
        AutoDiff           % Turn on/off automatic differentiation
        HFuntion           % Instance of function to compute h(theta)
        GradHFuntion       % Instance of function to compute gradient of h(theta)
        DataTrain          % Training data
        Setting            % Struct to store additional setting to the model
        SaveParams         % If save parameters in all iterations or not
        Optimization       % Optimization method
    end
    
    methods
        function obj = VBayesLab(varargin)
            %MODEL Construct an instance of this class
            %   Detailed explanation goes here
            obj.AutoDiff     = false;
            obj.GradientMax  = 100; 
            obj.GradWeight   = 0.9;
            obj.InitMethod   = 'Random';
            obj.LBPlot       = true;
            obj.LearningRate = 0.001;
            obj.MaxIter      = 5000;
            obj.MaxPatience  = 20;
            obj.NumSample    = 50;
            obj.StdForInit   = 0.01;
            obj.SigInitScale = 0.1;
            obj.StepAdaptive = obj.MaxIter/2;
            obj.SaveParams   = false;
            obj.Verbose      = true;
            obj.WindowSize   = 30;
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

