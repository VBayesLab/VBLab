function data_out = readData(dataName, varargin)

% Initialize additional options
Intercept = true;  
Length = NaN;
Normalized = true;
Index = '';
Type = 'Matrix';
RealizedMeasure = '';

% Load user's options
if nargin > 1
    %Parse additional options
    paramNames = {'Intercept'     'Length'     'Normalized' ...
                  'Index'         'Type'       'RealizedMeasure'};
    paramDflts = { Intercept      Length       Normalized ...
                   Index          Type         RealizedMeasure };

    [Intercept,...
     Length,...
     Normalized,...
     Index,...
     Type,...
     RealizedMeasure] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
end 

% Load built-in datasets
datatype = '';
switch dataName
    % Abalon data
    case 'Abalon'
        datatype = 'Cross-Sectional';
        data = load('Abalon.mat');
        data_mat = data.data;
        
    % DirectMarketing data
    case 'DirectMarketing'
        datatype = 'Cross-Sectional';
        data = load('DirectMarketing.mat');
        data_mat = data.data;
        if(Normalized)
            norm_col = [1,2,3,12];
            data_mat(:,norm_col) = zscore(data_mat(:,norm_col));
        end
        
    % GermanCredit data
    case 'GermanCredit'
        datatype = 'Cross-Sectional';
        data = load('GermanCredit.mat');
        data_mat = data.data;
        if(Normalized)
            data_mat = [zscore(data_mat(:,1:15)),data_mat(:,16:end)];
        end
        
    % LabourForce data
    case 'LabourForce'
        datatype = 'Cross-Sectional';
        data = load('LabourForce.mat');
        data_mat = data.data;
        if(Normalized)
            norm_col = [3,4,5,6];
            data_mat(:,norm_col) = (data_mat(:,norm_col)-mean(data_mat(:,norm_col)))./std(data_mat(:,norm_col));
        end       
        
    % RealizedLibrary data
    case 'RealizedLibrary'
        datatype = 'TimeSeries';
        data = load('RealizedLibrary.mat');
        % If length is specified
        if(isempty(Index))
            error('At least one index must be specified!')
        else
            data_mat = data.(Index).open_to_close*100;  
            if Length > 0 
                T = Length;
                if Length <= length(data_mat)
                    data_mat = data_mat(end-T+1:end);
                else
                    error('The Length argument must be smaller than the lenght of the time series!')
                end
            end
        end
        
        if(~isempty(RealizedMeasure))
            num_obs = length(data_mat);
            data_out.return = data_mat;
            if iscell(RealizedMeasure)
                num_realized = length(RealizedMeasure);
                for i = 1:num_realized
                    data_out.(RealizedMeasure{i}) = data.(Index).(RealizedMeasure{i})(end-num_obs+1:end)*100^2;
                end
            else
                data_out.(RealizedMeasure) = data.(Index).(RealizedMeasure)(end-num_obs+1:end)*100^2;    
            end
        else
            data_out = data_mat;
        end
end

%% Check additional options
% If a column of 1 is added to the matrix X of cross-sectional data (default)
if strcmp(datatype,'Cross-Sectional') && Intercept
    data_mat = [ones(size(data_mat,1),1),data_mat];
    VarNames = ['Intercept',data.VarNames];
    data_out = data_mat;
    if strcmp(Type,'Table')
        data_table = array2table(data_mat);
        data_table.Properties.VariableNames = VarNames;
        data_out = data_table;
    end
else
    data_out = data_mat;
end

end

