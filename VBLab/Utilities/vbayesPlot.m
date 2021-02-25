function vbayesPlot(type,value,varargin)
%VBAYESPLOT Plot analytics figures for results from VB or MCMC 
% Input: 
%     Type: string specifying type of plot, including
%           'Density': Plot density distribution. 
%           'Shrinkage': Plot shrinkage coefficients of a deepGLM model
%           'Interval': Plot prediction interval for continuous responses
%           'ROC': plot ROC curve for binary response. 'Ytest' must be provided
%
%     value: Additional information associated with each type
%           'Density' -> value is a cell array of distribution name and parameters
%           'Shrinkage' -> value is an NxD array of shrinkage parameters
%           'Interval' -> value is a 1D array of prediction values
%           'ROC' -> value iss a 1D array of prediction 


    if nargin < 2
        error(utils_errorMsg('vbayeslab:TooFewInputs'));
    end

    %% Parse additional options
    paramNames = {'Title'          'Xlabel'          'Ylabel'         'LineWidth',...
                  'Color'          'IntervalStyle'   'Nsample'        'Ordering',...
                  'yTest'          'Legend'          'Subplot'        'VarNames' };

    paramDflts = {NaN              NaN               NaN              2,...                 
                  'red'            'shade'           50               'ascend',...
                  NaN              NaN               NaN              NaN};

    [TextTitle,labelX,labelY,linewidth,...
     color,style,npoint,order,...
     yTest,Textlegend,VarNames] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

    % Store plot options to a structure
    opt.title = TextTitle;
    opt.labelX = labelX;
    opt.labelY = labelY;
    opt.linewidth = linewidth;
    opt.color = color;

    switch type
        % Plot distribution density
        % value must be a cell array with distribution name and
        % distribution parameters
        case 'Density'
            eval(['dist=',value{1},';']); % Use distribution name as a distribution object
            params = value{2};  % Distribution parameters
            dist.plotPdf(params);
                        
        % Plot shrinkage parameters of a deepGLM model 
        case 'Shrinkage'
            plotShrinkage(value,opt);
            
        % Plot prediction interval for continuous output
        case 'Interval'
            yhat = value.yhatMatrix;
            yhatInterval = value.interval;
            predMean = mean(yhat);
            % If test data have more than 100 rows, extract randomly 100 points to draw
            if(length(predMean)>=npoint)
                idx = randperm(length(yhatInterval),npoint);
                intervalPlot = yhatInterval(idx,:);
                yhatMeanPlot = predMean(idx)';
                if(~isempty(yTest))
                     ytruePlot = yTest(idx)';
                end
            else
                yhatMeanPlot = predMean';
                intervalPlot = yhatInterval;
                ytruePlot = yTest;
            end
            % Sort data
            [yhatMeanPlot,sortIdx] = sort(yhatMeanPlot,order);
            intervalPlot = intervalPlot(sortIdx,:);
            if(isempty(yTest))
                ytruePlot = [];
            else
                ytruePlot = ytruePlot(sortIdx);
            end
            plotInterval(yhatMeanPlot,intervalPlot,opt,...
                        'ytrue',ytruePlot,...
                        'Style',style);
                    
        % Plot ROC curve for binary outcomes
        % Value is prediction class labels. Could be a 1D array (single ROC)
        % or cell array of 1D array (multiple ROC)
        % The 'Ytest' argument must be provided
        case 'ROC'
            if(~isnumeric(yTest))
                disp('Target should be a column of binary responses!')
                return
            else
                % Plot single ROC
                if(size(value,2)==1)
                    [tpr,fpr,~] = roc(yTest',value');
                    plot(fpr,tpr,'LineWidth',linewidth);
                    grid on
                    title(TextTitle,'FontSize',20);
                    xlabel(labelX,'FontSize',15);
                    ylabel(labelY,'FontSize',15);
                % Plot multiple ROC
                else
                    tpr = cell(1,size(value,2));
                    fpr = cell(1,size(value,2));
                    for i=1:size(Pred,2)
                        [tpr{i},fpr{i},~] = roc(yTest',value(:,i)');
                        plot(fpr{i},tpr{i},'LineWidth',linewidth);
                        grid on
                        hold on
                    end
                    title(TextTitle,'FontSize',20);
                    xlabel(labelX,'FontSize',15);
                    ylabel(labelY,'FontSize',15);
                    legend(Textlegend{1},Textlegend{2});
                end
            end
    end
end

