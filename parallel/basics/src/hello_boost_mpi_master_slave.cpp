// STL
#include <cstdlib>
#include <iostream>
// boost::mpi
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/status.hpp>
namespace mpi = boost::mpi;
// Definitions
#define NUMBER_OF_JOBS 12

int main(int argc, char const *argv[]) {
  // Initialize the MPI environment
  mpi::environment env;
  mpi::communicator world;

  // Test world size
  if (world.size() > NUMBER_OF_JOBS  + 1) {
    if (world.rank() == 0) {
      std::cerr << "Too many processes (" << world.size() 
                <<  ") for the number of jobs!\n";
      std::cerr << "Use " << NUMBER_OF_JOBS + 1 << " ranks or less\n";
      return 0;  // Return 0 to avoid openMPI error messages
    }
    else {
      return 0;  // Return 0 to avoid openMPI error messages
    }
  }


  /********************************* MASTER **********************************/
  if (world.rank() == 0) {
    // Initialize requests
    unsigned int job_id = 0;
    std::vector<mpi::request> reqs(world.size());

    // Send initial jobs
    for (unsigned int dst_rank = 1; dst_rank < world.size(); ++dst_rank) {
      std::cout << "[MASTER] Sending job " << job_id
                << " to SLAVE " <<  dst_rank << "\n";
      // Send job to dst_rank [nonblocking]
      world.isend(dst_rank, 0, job_id);
      // Post receive request for new jobs requests by slave [nonblocking]
      reqs[dst_rank] = world.irecv(dst_rank, 0);
      ++job_id;
    }

    // Send jobs as long as there is job left
    while(job_id < NUMBER_OF_JOBS) {
      bool stop;
      for (unsigned int dst_rank = 1; dst_rank < world.size(); ++dst_rank) {
        // Check if dst_rank is done
        if (reqs[dst_rank].test()) {
          std::cout << "[MASTER] Rank " << dst_rank << " is done.\n";
          // Check if there is remaining jobs
          if (job_id  < NUMBER_OF_JOBS) {
            // Tell the slave that a new job is coming.
            stop = false;
            world.isend(dst_rank, 0, stop);
            // Send the new job.
            std::cout << "[MASTER] Sending new job (" << job_id
                      << ") to SLAVE " << dst_rank << ".\n";
            world.isend(dst_rank, 0, job_id);
            reqs[dst_rank] = world.irecv(dst_rank, 0);
            ++job_id;
          }
          else {
            // Send stop message to slave.
            stop = true;
            world.isend(dst_rank, 0, stop);
          }
        }
      }
      usleep(1000);
    }
    std::cout << "[MASTER] Sent all jobs.\n";

    // Listen for the remaining jobs, and send stop messages on completion.
    bool all_done = false;
    while (!all_done) {
      all_done = true;
      for (unsigned int dst_rank = 1; dst_rank < world.size(); ++dst_rank) {
        if (reqs[dst_rank].test()) {
            // Tell the slave that it can exit.
            bool stop = true;
            world.isend(dst_rank, 0, stop);
        }
        else {
          all_done = false;
        }
      }
      usleep(1000);
    }
    std::cout << "[MASTER] Handled all jobs, killed every process.\n";
  }


  /********************************* SLAVES **********************************/
  if (world.rank() != 0) {
    bool stop = false;
    while(!stop) {
      // Wait for new job
      unsigned int job_id = 0;
      world.recv(0, 0, job_id);
      std::cout << "[SLAVE: " << world.rank()
                << "] Received job " << job_id << " from MASTER.\n";
      // Perform "job"
      int sleep_time = std::rand()/100000;
      std::cout << "[SLAVE: "<< world.rank() 
                << "] Sleeping for " << sleep_time
                << " microseconds (job " << job_id << ").\n";
      usleep(sleep_time);
      // Notify master that the job is done
      std::cout << "[SLAVE: " << world.rank() 
                << "] Done with job " << job_id << ". Notifying MASTER.\n";
      world.send(0, 0);
      // Check if a new job is coming
      world.recv(0, 0, stop);
    }
  }

  std::cout << "~~~~~~~~ Rank " << world.rank() << " is exiting ~~~~~~~~~~~\n";
  return EXIT_SUCCESS;
}
