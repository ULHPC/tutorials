%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% example2.m :
%%%	Non-interactive script that shows:
%%%     - serial execution of time consuming operations
%%%     - parallel execution and relative speedup vs serial execution
%%%     - GPU-based parallel execution
%%%
%%%		Valentin Plugaru <Valentin.Plugaru@uni.lu> 2014-2019
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Generate a square matrix with normally distributed pseudorandom numbers
dim = 5000;
data = randn(dim);
maxiter = 10;

fprintf('-- Will perform %i iterations on a %ix%i matrix\n\n', maxiter, dim, dim);

% Serial execution of time consuming operations
data2 = data;
fprintf('-- Serial test\n');
tic
for i=1:maxiter
	  c(:,i) = eig(data2);
end
timings = toc;
timings_full = timings;
nprocs = 1;
fprintf('-- Execution time: %fs.\n', timings);


% Take number of parallel threads from SLURM job's #cores/node
maxWorkers = str2num(getenv('SLURM_CPUS_ON_NODE'));
fprintf('-- Parallel tests with up to %i cores\n', maxWorkers);
% Define a temporary location for files created by parcluster()
tmpJobStorageLocation = strcat(getenv('SCRATCH'),'matlab.',getenv('SLURM_JOB_ID'))

% Parallel execution of independent tasks
for nproc = 2:+2:maxWorkers
    data2 = data;
    mkdir(tmpJobStorageLocation); % create temporary folder
    looptime = tic;
    cluster = parcluster();
    cluster.JobStorageLocation = tmpJobStorageLocation;
    cluster.NumWorkers = nproc;
    fprintf('\n-- Parallel test using %i cores\n', cluster.NumWorkers);
    p = parpool(cluster);
    exectime = tic;
    parfor i=1:maxiter
        c(:,i) = eig(data2);
    end
    timings = [timings toc(exectime)];
    delete(p);
    timings_full = [timings_full toc(looptime)];
    nprocs = [nprocs nproc];
    fprintf('-- Execution time: %fs.\n', timings(end));
    fprintf('-- Execution time with overhead: %fs.\n', timings_full(end));
    rmdir(tmpJobStorageLocation, 's'); % cleanup temporary folder
end
fprintf('\n-- Number of processes, parallel execution time (s), parallel execution time with overhead(s), speedup, speedup with overhead:\n');
disp([nprocs' timings' timings_full' 1./(timings'/timings(1)) 1./(timings_full'/timings_full(1))]);
%plot(nprocs, timings)
%hold on
%plot(nprocs, timings_full)
%ylim([0 max([timings timings_full])])


% GPU-Parallel execution of independent tasks, if available
if gpuDeviceCount > 0
    fprintf('\n-- GPU test on %i GPU(s) \n', gpuDeviceCount)
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
    fprintf('\n-- GPUs not available on this system. Not running GPU-parallel test.\n')
end

% Explicitly call 'quit' in order to exit the Matlab session
% because of this, we can run matlab as:
%   matlab -nodisplay -nosplash -r example2
% (the missing '.m' extension is intentional!) instead of:
%   matlab -nodisplay -nospash < example2.m
quit
