%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% file_data_source.m :
%%%		Stock data retrieval from formatted CSV file
%%%		Valentin Plugaru <Valentin.Plugaru@uni.lu> 2018
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hist_date, hist_high, hist_low, hist_open, hist_close, hist_vol] = file_data_source(stock_symbol, verbose)

% Make 'verbose' parameter optional (default to false)
if (~exist('verbose', 'var'))
	verbose = false;
end

% Build file name
file_name = strcat(stock_symbol, '.csv')

if verbose
	sprintf('-- Reading file: %s', file_name)
end

% Read file with specific format 
formatSpec = '%{yyyy-MM-dd}D%f%f%f%f%f%d';
datatable = readtable(file_name, 'Format', formatSpec)

% Adjust for dividends, splits, etc.
DATEtemp  = datatable.Date
OPENtemp  = datatable.Open  .* datatable.AdjClose ./ datatable.Close;
HIGHtemp  = datatable.High  .* datatable.AdjClose ./ datatable.Close;
LOWtemp   = datatable.Low   .* datatable.AdjClose ./ datatable.Close;
CLOSEtemp = datatable.Close .* datatable.AdjClose ./ datatable.Close;
VOLtemp   = datatable.Volume;

if verbose
	sprintf('-- Retrieved data for %i days.', length(DATEtemp))
end

% If needed to reverse to achieve chronological order (1st entry is oldest data point), use flipud
hist_date  = DATEtemp;
hist_open  = OPENtemp;
hist_high  = HIGHtemp;
hist_low   = LOWtemp;
hist_close = CLOSEtemp;
hist_vol   = VOLtemp;
