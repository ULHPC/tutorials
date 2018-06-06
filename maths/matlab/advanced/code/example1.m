%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% example1.m :
%%%		Non-interactive script that shows:
%%%     - how to use an external function that retrieves financial data
%%%     - how to use different plotting methods
%%%     - how to export the plots in different graphic formats instead of 
%%%       displaying them
%%%
%%%		Valentin Plugaru <Valentin.Plugaru@gmail.com> 2014-03-18
%%%     - 2017-06-12: Use Google Finance in place of Yahoo, which dissapeared
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start a stopwatch timer we'll use to see how much time this script takes
% to run
tic

% Set company ticker, e.g. 'AAPL' for Apple
company_ticker = 'AAPL';

% Use the external function to download Google finance data for the selected period
[hist_date, hist_high, hist_low, hist_open, hist_close, hist_vol] = google_finance_data(company_ticker, '2016-01-01', '2017-01-01');

% Convert string dates for plotting
dates = datenum(hist_date);

% 2D Plot: dates on X axis and closing prices on Y axis
f = figure('visible','off');
plot(dates, hist_close);

% Set format for X axis tickers and labels/titles
dateaxis('x', 17);
xlabel('Date');
ylabel('Price (USD)');
title(sprintf('Closing stock prices for %s between %s and %s ', company_ticker, hist_date{1}, hist_date{end}));

% Save plot to PDF
saveas(f, 'example1-2dplot.pdf');

% Also save plot to black and white EPS, with 300 DPI
dateaxis('x', 6);
print(f, '-deps', '-r300', 'example1-2dplot.eps');

% 3D Scatter plot: dates on X axis, closing prices on Y axis, volume of shares on Z axis
% Furthermore, print in a different color datapoints where the closing
% share price was above 100 (hm, in 2014 I did the split at 500..)
f = figure('visible','off');
scatter3(dates, hist_close, hist_vol, [] , hist_close > 100);

% Set format for X axis tickers and labels/titles
dateaxis('x', 17);
xlabel('Date');
ylabel('Price (USD)');
zlabel('Trading volume');
title(sprintf('Closing stock prices and trading volumes for %s between %s and %s ', company_ticker, hist_date{1}, hist_date{end}));

% Save plot to color EPS, with 300 DPI
print(f, '-depsc', '-r300', 'example1-scatter.eps');

% Show how much time everything took
toc
