%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% google_finance_data.m :
%%%		Google Finance stock data retrieval for Matlab
%%%		Valentin Plugaru <Valentin.Plugaru@gmail.com> 2014-03-18
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [hist_date, hist_high, hist_low, hist_open, hist_close, hist_vol] = yahoo_finance_data(stock_symbol, s_date, e_date, verbose)

% Make 'verbose' parameter optional (default to false)
if (~exist('verbose', 'var'))
	verbose = false;
end
% http://www.google.com/finance/historical?q=AAPL&startdate=2017-01-01&output=csv
% Build URL string, month indexing starts from 0
url_string = 'http://www.google.com/finance/historical?';
url_string = strcat(url_string, '&q=', upper(stock_symbol));
url_string = strcat(url_string, '&startdate=', s_date );
url_string = strcat(url_string, '&enddate=', e_date );
url_string = strcat(url_string, '&output=csv');

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
    %sprintf('%s', buff_line) 
    % Break if this is the end
    if length(buff_line) < 6, break; end
    
	csvdata = strsplit(buff_line, ',');

    % Extract high, low, open, close, etc. from string
    DATEvar   = csvdata{1};
    OPENvar   = str2num( csvdata{2} );
    HIGHvar   = str2num( csvdata{3} );
    LOWvar    = str2num( csvdata{4} );
    CLOSEvar  = str2num( csvdata{5} );
    VOLvar    = str2num( csvdata{6} );

    DATEtemp{ptr,1} = DATEvar;
    OPENtemp(ptr,1) = OPENvar;
    HIGHtemp(ptr,1) = HIGHvar;
    LOWtemp (ptr,1) = LOWvar;
    CLOSEtemp(ptr,1)= CLOSEvar;
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
