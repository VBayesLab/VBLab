function data = readData(dataName, varargin)

% Initialize additional options
Intercept = true;  
Length = 0;
Normalized = true;

% Load user's options
if nargin > 1
    %Parse additional options
    paramNames = {'Intercept'     'Length'     'Normalized'};
    paramDflts = { Intercept      Length       Normalized};

    [Intercept,...
     Length,...
     Normalized] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
end 

% Load built-in datasets
datatype = '';
switch dataName
    case 'LabourForce'
        datatype = 'Cross-Sectional';
        data = load('LabourForce.mat');
        
    case 'GermanCredit'
        datatype = 'Cross-Sectional';
        data = load('GermanCredit.mat');
        if (Normalized)
            data.X = [zscore(data.X(:,1:15)),data.X(:,16:end)];
        end
        
    case 'SP500'
        datatype = 'TimeSeries';
        data = load('StockIndex.mat');
        
        if Length > 0
            
        end
end

%% Check additional options
% If a column of 1 is added to the matrix X of cross-sectional data (default)
if strcmp(datatype,'Cross-Sectional') && Intercept
    data.X = [ones(size(data.X,1),1),data.X];
end


end

