%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% yahoo_finance_data.m :
%%%		Yahoo! Finance stock data retrieval for Matlab
%%%		Valentin Plugaru <Valentin.Plugaru@gmail.com> 2014-03-18
%%%
%%% Adapted from: LuminousLogic.com - get_hist_stock_data.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hist_date, hist_high, hist_low, hist_open, hist_close, hist_vol] = yahoo_finance_data(stock_symbol, s_year, s_month, s_day, e_year, e_month, e_day, verbose)

% Make 'verbose' parameter optional (default to false)
if (~exist('verbose', 'var'))
	verbose = false;
end

% Build URL string, month indexing starts from 0
url_string = 'http://ichart.finance.yahoo.com/table.csv?';
url_string = strcat(url_string, '&s=', upper(stock_symbol));
url_string = strcat(url_string, '&a=', num2str(s_month-1) );
url_string = strcat(url_string, '&b=', num2str(s_day)     );
url_string = strcat(url_string, '&c=', num2str(s_year)    );
url_string = strcat(url_string, '&d=', num2str(e_month-1) );
url_string = strcat(url_string, '&e=', num2str(e_day)     );
url_string = strcat(url_string, '&f=', num2str(e_year)    );
url_string = strcat(url_string, '&g=d&ignore=.csv');

if verbose
	sprintf('-- Request URL: %s', url_string)
end

% Open a connection to the URL and retrieve data into a buffer
buffer      = java.io.BufferedReader(...
              java.io.InputStreamReader(...
              openStream(...
              java.net.URL(url_string))));

% Read the first line (a header) and discard
dummy   = readLine(buffer);

% Read all remaining lines in buffer
ptr = 1;
while 1
    % Read line
    buff_line = char(readLine(buffer)); 
    
    % Break if this is the end
    if length(buff_line)<3, break; end
    
	csvdata = strsplit(buff_line, ',');

    % Extract high, low, open, close, etc. from string
    DATEvar   = csvdata{1};
    OPENvar   = str2num( csvdata{2} );
    HIGHvar   = str2num( csvdata{3} );
    LOWvar    = str2num( csvdata{4} );
    CLOSEvar  = str2num( csvdata{5} );
    VOLvar    = str2num( csvdata{6} );
    adj_close = str2num( csvdata{7} );

    %Adjust for dividends, splits, etc.
    DATEtemp{ptr,1} = DATEvar;
    OPENtemp(ptr,1) = OPENvar  * adj_close / CLOSEvar;
    HIGHtemp(ptr,1) = HIGHvar  * adj_close / CLOSEvar;
    LOWtemp (ptr,1) = LOWvar   * adj_close / CLOSEvar;
    CLOSEtemp(ptr,1)= CLOSEvar * adj_close / CLOSEvar;
    VOLtemp(ptr,1)  = VOLvar;

    ptr = ptr + 1;
end

if verbose
	sprintf('-- Retrieved data for %i days.', length(DATEtemp))
end

% Reverse to normal chronological order, so 1st entry is oldest data point
hist_date  = flipud(DATEtemp);
hist_open  = flipud(OPENtemp);
hist_high  = flipud(HIGHtemp);
hist_low   = flipud(LOWtemp);
hist_close = flipud(CLOSEtemp);
hist_vol   = flipud(VOLtemp);
