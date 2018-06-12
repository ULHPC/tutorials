%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% example2.m :
%%%		Non-interactive script that shows:
%%%     - serial execution of time consuming operations
%%%     - parallel execution and relative speedup vs serial execution, setting
%%%       the maximum number of parallel threads through environment variables
%%%     - GPU-based parallel execution
%%%
%%%		Valentin Plugaru <Valentin.Plugaru@gmail.com> 2014-03-18
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Generate a square matrix with normally distributed pseudorandom numbers
dim = 10^3;
data = randn(dim);
maxiter = 24;

fprintf('-- Will perform %i iterations on a %ix%i matrix\n\n', maxiter, dim, dim)

% Serial execution of time consuming operations
data2 = data;
fprintf('-- Serial test\n')
tic
for i=1:maxiter
	  c(:,i) = eig(data2);
end
timings = toc;
timings_full = timings;
nprocs = 1;
fprintf('-- Execution time: %fs.\n', timings)

% Parallel execution of independent tasks

% Set a default maximum number of parallel threads
maxWorkers = 4;

% Override maxWorkers with user's (Linux) environment MATLABMP variable
% (e.g. set with 'export MATLABMP=12' before Matlab's execution)
envMaxWorkers = getenv('MATLABMP');
if ~isempty(envMaxWorkers)
    maxWorkers = str2num(envMaxWorkers);
    fprintf('\n-- Found environment variable MATLABMP=%i.\n',maxWorkers)
end
fprintf('-- Parallel tests with up to %i cores\n', maxWorkers)

for nproc = 2:maxWorkers
    data2 = data;
    looptime = tic;
    cluster = parcluster();
    cluster.NumWorkers = nproc;
    fprintf('\n-- Parallel test using %i cores\n', cluster.NumWorkers)
    myparpool = parpool(cluster);
    exectime = tic;
    parfor i=1:maxiter
        c(:,i) = eig(data2);
    end
    timings = [timings toc(exectime)];
    delete(myparpool);
    timings_full = [timings_full toc(looptime)];
    nprocs = [nprocs nproc];
    fprintf('-- Execution time: %fs.\n', timings(end))
    fprintf('-- Execution time with overhead: %fs.\n', timings_full(end))
end
fprintf('\n-- Number of processes, parallel execution time (s), parallel execution time with overhead(s), speedup, speedup with overhead:\n')
disp([nprocs' timings' timings_full' 1./(timings'/timings(1)) 1./(timings_full'/timings_full(1))])
%plot(nprocs, timings)
%hold on
%plot(nprocs, timings_full)
%ylim([0 max([timings timings_full])])



% GPU-Parallel execution of independent tasks, if available

if gpuDeviceCount > 0
    fprintf('\n-- GPU test \n')
    looptime_gpu = tic;
    data2 = gpuArray(data);
    exectime_gpu = tic;
    for i=1:maxiter
       eig(data2);
    end
    timings_gpu = toc(exectime_gpu);
    timings_full_gpu = toc(looptime_gpu);
    fprintf('-- GPU Execution time: %fs.\n', timings_gpu)
    fprintf('-- GPU Execution time with overhead: %fs.\n', timings_full_gpu)
    fprintf('-- GPU vs Serial speedup: %f.\n', timings(1)/timings_gpu)
    fprintf('-- GPU with overhead vs Serial speedup: %f.\n', timings_full(1)/timings_full_gpu)
else
    fprintf('\n-- GPU-Parallel test not available on this system.\n')
end

% Explicitly call 'quit' in order to exit the Matlab session
% because of this, we can run matlab as:
%   matlab -nodisplay -nosplash -r example2
% (the missing '.m' extension is intentional!) instead of:
%   matlab -nodisplay -nospash < example2.m
quit
