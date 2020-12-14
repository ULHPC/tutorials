/****************************************************************************
Copyright (c) 2016, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/
#include <iostream>
#include <vector>
#include <algorithm>
#include "luo_rudy_1991.hpp"
#include "fhn.hpp"
#include <pthread.h>
#include <getopt.h>
#include <sstream>
#include <numeric>
#include <fcntl.h>
#include <sys/stat.h>
#include <iterator>
#include <map>
#include <mpi.h>
#include <omp.h>

#if defined (ENABLE_GL)
#include <QApplication>
#include <QDesktopWidget>
#include "window.h"
#endif

#include <math.h>
#include "mesh.hpp"
#include <assert.h>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits>
extern void genrcm ( int node_num, int adj_num, int adj_row[], int adj[], int perm[] );
struct options {
    std::string mesh_file;
    std::string setups_file;
    bool need_visualize;
    double solve_time;
    bool improve_locality;
    bool block_gather;
    bool save_images;
    std::string read_state;
    std::string write_state;
} options;
// typedef LR_I DynamicalSystem;

#define FHN_MODE 0
#define LR_I_MODE 1

#define SYS_MODE LR_I_MODE

#if defined(SYS_MODE) && (SYS_MODE == LR_I_MODE)
typedef LR_I  DynamicalSystem;
#define DATA_RANGE_MIN (-90.0)
#define DATA_RANGE_MAX (40.0)
#elif defined(SYS_MODE) && (SYS_MODE == FHN_MODE)
typedef FHN  DynamicalSystem;
#define DATA_RANGE_MIN (-2.0)
#define DATA_RANGE_MAX (2.0)
#else
#error No/incorrect dynamical system mode is defined
#endif


#define COLL_PRINT(comm, format,  ...) do {                              \
        int _s, _r;                                                     \
        MPI_Comm_size(comm, &_s);                             \
        MPI_Comm_rank(comm, &_r);                             \
        for (int i=0; i<_s; i++) {                                      \
            if (i == _r) {                                              \
                printf("rank %d: " format "\n",_r, ## __VA_ARGS__);       \
                fflush(stdout);                                         \
                usleep(10000);                                          \
            }                                                           \
            MPI_Barrier(comm);                                \
        }                                                               \
    }while(0);

#define ROOT_PRINT(comm, format, ...) do {                       \
        int _s, _r;                                             \
        MPI_Comm_rank(comm, &_r);                     \
        if (0 == _r) {                                          \
            printf(format "\n", ## __VA_ARGS__);   \
            fflush(stdout);                                     \
            usleep(10000);                                      \
        }                                                       \
    }while(0);


static inline double
dist(float *c1, float *c2) {
    return (double)sqrt((c1[0]-c2[0])*(c1[0]-c2[0])+
                (c1[1]-c2[1])*(c1[1]-c2[1])+
                (c1[2]-c2[2])*(c1[2]-c2[2]));
}

typedef struct compute_node compute_node_t;
#define MAX_NEIGHBOURS 32
struct compute_node{
    int vtkPointId;
    DynamicalSystem cell;
    double state[DynamicalSystem::SYS_SIZE];
    double rk4[4][DynamicalSystem::SYS_SIZE];
    int n_ids[MAX_NEIGHBOURS];      // neighbours_ids
    int n_num;
    int remote_n_num;
    double n_dists[MAX_NEIGHBOURS]; //neighbours distances
};

static inline void
init_adjacency(std::vector<compute_node_t> &cnodes,
               int &node_num, int &adj_num, std::vector<int> &adj_row,
               std::vector<int> &adj, std::vector<int> &perm) {
    node_num = cnodes.size();
    adj_row.resize(node_num+1);
    adj_num = 0;
    for (auto &n : cnodes)
        adj_num += n.n_num;
    adj.resize(adj_num);
    perm.resize(node_num);
    auto adj_it = adj.begin();
    int offset = 1;

    for (int i=0; i<node_num; i++) {
        for (int j=0; j<cnodes[i].n_num; j++)
            *adj_it++ = cnodes[i].n_ids[j];

        adj_row[i] = offset;
        offset += cnodes[i].n_num;
    }
    adj_row[node_num] = offset;
    for (auto &v : adj) v+=1;

}

static inline void
improve_locality(std::vector<compute_node_t> &cnodes) {
    int node_num;
    int adj_num;
    std::vector<int> adj_row;
    std::vector<int> adj;
    std::vector<int> perm;
    init_adjacency(cnodes, node_num, adj_num,
                   adj_row, adj, perm);
    genrcm(node_num, adj_num,
           adj_row.data(),
           adj.data(),
           perm.data());

    // std::reverse(perm.begin(),perm.end());
    for (auto &v : perm) v-=1;
    std::vector<int> tmp(node_num);
    for (int i=0; i<node_num; i++)
        tmp[perm[i]]=i;
    std::vector<compute_node_t> improved(node_num);
    for (int i=0; i<node_num; i++) {
        improved[i] = cnodes[perm[i]];
        for (int j=0; j<cnodes[perm[i]].n_num; j++)
            improved[i].n_ids[j] = tmp[cnodes[perm[i]].n_ids[j]];
    }
    improved.swap(cnodes);
}

class Task {
private:
    int node_number;
    int elem_number;
    Mesh *mesh;
    std::vector<compute_node_t> cnodes;
    double D;
    double dt;
    bool vizualize;
    int mpi_rank;
    std::vector<double> non_local_values;
    std::map<int, std::vector<int> > non_loc_dependecies;
    std::map<int, std::vector<int> > non_loc_dependees;
    std::map<int, std::vector<double> > non_loc_dependees_bufs;
    std::vector<MPI_Request> sreqs;
    std::vector<MPI_Request> rreqs;
    std::vector<MPI_Status> sstats;
    std::vector<MPI_Status> rstats;
    std::vector<int> vtkIdsOrder;
    std::vector<float> localVizBuf;
    // significant at viz rank only
    int viz_rank;
    MPI_Comm work_comm;
    int work_size;
    int work_rank;
    int world_size;
    MPI_Request gather_req;;
    int gather_completed;
    int coll_count;
    int start_time;
public:
    void setWorkComm(MPI_Comm comm) {
        work_comm = comm;
        MPI_Comm_size(work_comm,&work_size);
        MPI_Comm_rank(work_comm,&work_rank);
    }
    inline void init_send_bufs(const int rk4_id) {
        // THIS HAS TO BE DONE SINGLETHREAD
        for (auto &d : non_loc_dependees) {
            int remote_rank = d.first;
            std::vector<int> &ids_to_pack = d.second;
            std::vector<double> &buf = non_loc_dependees_bufs[remote_rank];
#pragma omp for
            for (int i=0; i<ids_to_pack.size(); i++) {
                int j = ids_to_pack[i];
                buf[i] = cnodes[j].state[DynamicalSystem::COUPLING_VAR_ID];
                if (rk4_id == 3)
                    buf[i] += cnodes[j].rk4[2][DynamicalSystem::COUPLING_VAR_ID]/2.0;
                else if (rk4_id > 0)
                    buf[i] += cnodes[j].rk4[rk4_id-1][DynamicalSystem::COUPLING_VAR_ID];
            }
        }
    }

    inline void start_sends() {
        int count = 0;
        for (auto &d : non_loc_dependees ) {
            int remote_rank = d.first;
            double *buf = non_loc_dependees_bufs[remote_rank].data();
            int size = non_loc_dependees_bufs[remote_rank].size()*sizeof(double);
            MPI_Isend(buf,size,MPI_BYTE,remote_rank,123,
                      work_comm,&sreqs[count++]);

        }
    }

    inline void start_recvs() {
        int count = 0;
        int offset = 0;
        char *recv_buf = (char*)non_local_values.data();
        for (auto &d : non_loc_dependecies ) {
            int remote_rank = d.first;
            int size = d.second.size()*sizeof(double);
            MPI_Irecv(recv_buf+offset,size,MPI_BYTE,
                      remote_rank,123,work_comm,
                      &rreqs[count++]);
            offset += size;
        }
    }

    inline void wait_sends() {
        MPI_Waitall(sreqs.size(),sreqs.data(),sstats.data());
    }
    inline void wait_recvs() {
#pragma omp master
        MPI_Waitall(rreqs.size(),rreqs.data(),rstats.data());
#pragma omp barrier
    }

    inline void
    update_coupling_v2(std::vector<compute_node_t> &cnodes, const int rk4_id,
                       const double dt, const double D, const int N) {
        assert(cnodes.size() == N);
#pragma omp for
        for (int i=0; i<N; i++) {
            compute_node_t &node = cnodes[i];
            double C = 0;
            for (int j=0; j<node.n_num; j++) {
                int n_id = node.n_ids[j];
                double h = node.n_dists[j];
                compute_node_t &neighbour = cnodes[n_id];
                double v_neighbour;

                if (rk4_id == 0)
                    v_neighbour = neighbour.state[DynamicalSystem::COUPLING_VAR_ID];
                else if (rk4_id == 3)
                    v_neighbour = neighbour.state[DynamicalSystem::COUPLING_VAR_ID] + neighbour.rk4[rk4_id - 1][DynamicalSystem::COUPLING_VAR_ID];
                else
                    v_neighbour = neighbour.state[DynamicalSystem::COUPLING_VAR_ID] + neighbour.rk4[rk4_id - 1][DynamicalSystem::COUPLING_VAR_ID]/2.0;

                C += D*v_neighbour*h;
            }
            cnodes[i].rk4[rk4_id][DynamicalSystem::COUPLING_VAR_ID] +=
                dt*C;
        }
    }

    inline void
    update_coupling_remotes(std::vector<compute_node_t> &cnodes, const int rk4_id,
                            const double dt, const double D, const int N,
                            std::vector<double> &remote_values) {
#pragma omp for
        for (int i=0; i<N; i++) {
            compute_node_t &node = cnodes[i];
            double C = 0;
            for (int j=node.n_num; j<node.n_num+node.remote_n_num; j++) {
                int n_id = node.n_ids[j];
                double h = node.n_dists[j];
                double v_neighbour = remote_values[n_id];
                C += D*v_neighbour*h;
            }
            cnodes[i].rk4[rk4_id][DynamicalSystem::COUPLING_VAR_ID] +=
                dt*C;
        }
    }

    inline void exchange(int rk4_id) {
#pragma omp master
        {
            start_recvs();
        }
            init_send_bufs(rk4_id);
#pragma omp master
        {
            start_sends();
            wait_sends();
            // wait_recvs();
        }
    }
    inline void
    make_rk_step(std::vector<compute_node_t> &cnodes, int N, double dt, double D, double time) {

#pragma omp for
        for (int i=0; i<N; i++) {
            // if (fabs(cnodes[i].cell.Y[DynamicalSystem::COUPLING_VAR_ID]) > 200) {
                // fprintf(stderr,"attach: i=%d, V=%g, |V|=%g\n",i, cnodes[i].cell.Y[DynamicalSystem::COUPLING_VAR_ID],
                    // fabs(cnodes[i].cell.Y[DynamicalSystem::COUPLING_VAR_ID]));
                // volatile int flag=1;
                // while(flag) ;
            // }
            memcpy((void*)cnodes[i].state,cnodes[i].cell.Y,DynamicalSystem::SYS_SIZE*sizeof(double));
            cnodes[i].cell.compute(time,(double *)cnodes[i].rk4[0]);
            for (int j=0; j<DynamicalSystem::SYS_SIZE; j++)
                cnodes[i].rk4[0][j] *= dt;
        }

        exchange(0);
        update_coupling_v2(cnodes, 0, dt, D, N);
        wait_recvs();

        update_coupling_remotes(cnodes, 0, dt, D, N, non_local_values);

#pragma omp for
        for (int i=0; i<N; i++) {
            for (int j=0; j<DynamicalSystem::SYS_SIZE; j++)
                cnodes[i].cell.Y[j] = cnodes[i].state[j] + cnodes[i].rk4[0][j]/2.0;

            cnodes[i].cell.compute(time,cnodes[i].rk4[1]);
            for (int j=0; j<DynamicalSystem::SYS_SIZE; j++)
                cnodes[i].rk4[1][j] *= dt;
        }
        exchange(1);
        update_coupling_v2(cnodes, 1, dt, D, N);
        wait_recvs();

        update_coupling_remotes(cnodes, 1, dt, D, N, non_local_values);
#pragma omp for
        for (int i=0; i<N; i++) {
            for (int j=0; j<DynamicalSystem::SYS_SIZE; j++)
                cnodes[i].cell.Y[j] = cnodes[i].state[j] + cnodes[i].rk4[1][j]/2.0;

            cnodes[i].cell.compute(time,cnodes[i].rk4[2]);
            for (int j=0; j<DynamicalSystem::SYS_SIZE; j++)
                cnodes[i].rk4[2][j] *= dt;
        }
        exchange(2);
        update_coupling_v2(cnodes, 2, dt, D, N);
        wait_recvs();

        update_coupling_remotes(cnodes, 2, dt, D, N, non_local_values);
#pragma omp for
        for (int i=0; i<N; i++) {
            for (int j=0; j<DynamicalSystem::SYS_SIZE; j++)
                cnodes[i].cell.Y[j] = cnodes[i].state[j] + cnodes[i].rk4[2][j];

            cnodes[i].cell.compute(time,cnodes[i].rk4[3]);
            for (int j=0; j<DynamicalSystem::SYS_SIZE; j++)
                cnodes[i].rk4[3][j] *= dt;
        }
        exchange(3);
        update_coupling_v2(cnodes, 3, dt, D, N);
        wait_recvs();

        update_coupling_remotes(cnodes, 3, dt, D, N, non_local_values);

#pragma omp for
        for (int i=0; i<N; i++) {
            for (int j=0; j<DynamicalSystem::SYS_SIZE; j++)
                cnodes[i].cell.Y[j] = cnodes[i].state[j] +
                    (cnodes[i].rk4[0][j]+2.0*cnodes[i].rk4[1][j]+
                     2.0*cnodes[i].rk4[2][j]+cnodes[i].rk4[3][j])/6.0;

        }

    }
    void read_setups(float *d, float *pc, float *pr, double *dt) {
        std::ifstream ifs;
        ifs.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
        try{
            ifs.open(options.setups_file.c_str(), std::ios::in);
        }
        catch(std::ifstream::failure e){
            std::cout << "***Error: Can not open input file" <<
                options.setups_file << ": " << e.what() << std::endl;
        }
        string str[4];
        for (int i=0; i<4; i++) {
            ifs.ignore(numeric_limits<streamsize>::max(), '\n');
            std::getline(ifs,str[i]);
            ifs.ignore(numeric_limits<streamsize>::max(), '\n');
        }
        sscanf(str[0].c_str(),"%g",d);
        sscanf(str[1].c_str(),"%g %g %g",&pc[0],&pc[1],&pc[2]);
        sscanf(str[2].c_str(),"%g",pr);
        sscanf(str[3].c_str(),"%lf",dt);

    }
    Task() {
        mesh = NULL;
        D = 0;
        if (options.setups_file.size()) {
            float _d;
            float pc[3];
            float pr;
            read_setups(&_d, pc, &pr, &dt);
            D = _d;
        }

        start_time = 0;
        vizualize = false;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }
    Task(Mesh *_mesh,
         bool _visualize) {
        float paced_center[3] = {0, 0, 0};
        double radius = 1;//10//0.16;
        start_time = 0;
        D = 0;
        if (options.setups_file.size()) {
            float _d;
            float pc[3];
            float pr;
            read_setups(&_d, pc, &pr, &dt);
            D = _d;
            for (int i=0; i<3; i++)
                paced_center[i] = (float)pc[i];
            radius = pr;
        }

        vizualize = _visualize;
        mesh = _mesh;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        node_number = mesh->getPoints().size();
        elem_number = mesh->getTets().size();
        std::cout << "Number of Nodes: " << node_number << "\n" <<
            "Number of Cells: " << elem_number << std::endl;
        cnodes.resize(node_number);
        auto points = mesh->getPoints();
        for (auto i=0; i<node_number; i++) {
            int id = points[i].id;
            cnodes[id].vtkPointId = id;
            cnodes[id].cell.init();
            if (points[i].neighbours.size() > 0) {
                cnodes[id].n_num = points[i].neighbours.size();
                assert(cnodes[id].n_num >= 3);

                if (cnodes[id].n_num > MAX_NEIGHBOURS)
                    abort();

                for (int j=0; j<points[i].neighbours.size(); j++)
                    cnodes[id].n_ids[j] = points[i].neighbours[j];
            }else{
                cnodes[id].n_num = 0;
            }
        }

        for (auto i=0; i<node_number; i++) {
            auto &p = mesh->getPoints().at(cnodes[i].vtkPointId);
            double lumped_mass = 0;
            for (auto &m : p.mass)
                lumped_mass += m;
            for (auto j=0; j<cnodes[i].n_num; j++) {
                cnodes[i].n_dists[j] = -p.stiff[j]/lumped_mass;
#if 0
                auto c = mesh->getPoints().at(cnodes[i].vtkPointId).c;
                if (c[0] > mesh->getXcenter()) {
                    cnodes[i].n_dists[j] = 0;
                }
#endif
            }
        }

        for (int i=0; i<node_number; i++) {
            double d = dist(mesh->getPoints().at(cnodes[i].vtkPointId).c, paced_center);
            if (d < radius) {
#if SYS_MODE == LR_I_MODE
                cnodes[i].cell.stim_amplitude = -25.5;
                cnodes[i].cell.stim_period = 150;
                cnodes[i].cell.stim_start = 0;

#elif SYS_MODE == FHN_MODE
                cnodes[i].cell.a = 0;
#endif
            }
        }


        if (options.improve_locality)
            improve_locality(cnodes);

    }

    ~Task() { }

    void setD(double _D) {
        D = _D;
    }

    void solve(double TimeToSolve) {
        double time = start_time;
        int percents_done = 0;
        int progress_percent_step = 1;
        int count = 0;
#if defined(SYS_MODE) && (SYS_MODE == LR_I_MODE)
        double draw_time_step = 1; // redraw each 1 ms
#elif defined(SYS_MODE) && (SYS_MODE == FHN_MODE)
        double draw_time_step = 2; // redraw each 1 ms
#else
#error No/incorrect dynamical system mode is defined
#endif


        double next_draw_time = draw_time_step;
        double solve_speed;
        double t_start = omp_get_wtime();
        coll_count=0;
        TimeToSolve += start_time;
#pragma omp parallel private(time)
        {
#pragma omp master
            {
                fprintf(stderr,"RANK %d is using %d OMP threads\n",
                        mpi_rank, omp_get_num_threads());
            }

            int local_node_num = cnodes.size();

            while (time < TimeToSolve) {
                make_rk_step(cnodes, local_node_num,dt,D,time);
                time += dt;

#pragma omp master
                {
                    if (mpi_rank == 0) {
                        if (time / TimeToSolve * 100 > percents_done) {
                            solve_speed = time/(omp_get_wtime()-t_start);
                            std::cout << "  progress: " <<
                                percents_done << "%, current solution speed: " <<
                                solve_speed << " [model_ms/real_s] ...\r" << std::flush;
                            percents_done += progress_percent_step;
                        }
                    }


                    if (vizualize && time > next_draw_time) {
                        collectDataForVisualization();
                        next_draw_time += draw_time_step;
                    }

                }
            }
        }

        if (mpi_rank == 0) {
            std::cout << "  progress: " <<
                percents_done - progress_percent_step <<
                "%                                 " <<
                "                                \n" <<
                "solution speed: " <<
                solve_speed << " [model_ms/real_s]\n" <<
                "wall time: " << omp_get_wtime() - t_start << std::endl;
            if (options.need_visualize) {
                MPI_Send(&coll_count,1,MPI_INT,world_size-1,
                         999,MPI_COMM_WORLD);
            }
            start_time = time;
        }
    }

    double get_dt(){ return dt; }
    void setVisualize(bool value) { vizualize = value; }
    void setVizRank(int _v_rank) { viz_rank = _v_rank; }
    void collectDataForVisualization() {
        // SINGLE THREAD ONLY
        MPI_Status st;
        int sum;
        if (!options.block_gather) {
            if (gather_completed == -1) {
                sum = work_size;
                gather_completed = 0;
            }
            else{
                MPI_Test(&gather_req, &gather_completed, &st);
                MPI_Allreduce(&gather_completed,&sum,1,MPI_INT,MPI_SUM,
                              work_comm);
            }
        }
        if (options.block_gather || (sum == work_size)) {
            for (int i=0; i<cnodes.size(); i++)
                localVizBuf[i] = (float)cnodes[i].cell.get_var(DynamicalSystem::COUPLING_VAR_ID);

            MPI_Igatherv(localVizBuf.data(), localVizBuf.size(), MPI_FLOAT,
                         NULL,NULL,NULL,
                         MPI_FLOAT,world_size-1, MPI_COMM_WORLD, &gather_req);
            if (options.block_gather)
                MPI_Wait(&gather_req,&st);
            coll_count++;
        }

    }
    void distribute() {
        int r_start, r_len;
        int rank, size;
        gather_completed = -1;
        MPI_Comm_rank(work_comm, &rank);

        mpi_rank = rank;
        MPI_Comm_size(work_comm, &size);
        MPI_Bcast(&node_number, 1, MPI_INT, 0, work_comm);
        r_len = node_number / size;
        r_start = rank*r_len;
        if (rank < (node_number % size)){
            r_len++;
            r_start += rank;
        }else{
            r_start += (node_number % size);
        }
        COLL_PRINT(work_comm, "r_start=%d r_end=%d r_len=%d",
                   r_start, r_start+r_len-1, r_len);

        if (rank == 0) {
            std::vector<compute_node_t> cnodes_local(r_len);
            std::vector<int> r_lens(size);
            MPI_Gather(&r_len,1,MPI_INT,
                       r_lens.data(), 1, MPI_INT,
                       0, work_comm);

            std::vector<int> displs(size);

            displs[0] = 0;
            int cnode_size = sizeof(compute_node_t);
            for (auto &r : r_lens) r *= cnode_size;
            for (int i=1; i<size; i++) {
                displs[i] = displs[i-1]+r_lens[i-1];
            }
            MPI_Scatterv(cnodes.data(), r_lens.data(), displs.data(),
                         MPI_BYTE, cnodes_local.data(), r_len*cnode_size,
                         MPI_BYTE, 0, work_comm);
            if (vizualize) {

                vtkIdsOrder.resize(cnodes.size());
                for (int i=0; i<cnodes.size(); i++) {
                    vtkIdsOrder[i] = cnodes[i].vtkPointId;
                    if (!options.improve_locality)
                        assert(vtkIdsOrder[i] == i);
                }


                MPI_Send(vtkIdsOrder.data(), cnodes.size(), MPI_INT,
                         world_size-1, 1234, MPI_COMM_WORLD);
            }
            cnodes.swap(cnodes_local);
        } else {
            cnodes.resize(r_len);
            MPI_Gather(&r_len,1,MPI_INT,
                       NULL, 0, MPI_INT,
                       0, work_comm);
            MPI_Scatterv(NULL, NULL, NULL,
                         MPI_BYTE, cnodes.data(), r_len*sizeof(compute_node_t),
                         MPI_BYTE, 0, work_comm);
        }
        localVizBuf.resize(r_len);
        std::vector<int> non_local_ids;
        for (auto &c : cnodes) {
            std::vector<int> tmp(c.n_num);
            auto it = tmp.begin();
            std::vector<int> remote_n_ids;
            std::vector<double> remote_n_dists;
            std::vector<double> dists(c.n_num);
            auto d_it = dists.begin();
            c.remote_n_num = 0;
            for (int i=0; i<c.n_num; i++) {
                int n_id = c.n_ids[i];
                if (!(n_id >= r_start && n_id < r_start+r_len)) {
                    non_local_ids.push_back(n_id);
                    c.remote_n_num++;
                    remote_n_ids.push_back(n_id);
                    remote_n_dists.push_back(c.n_dists[i]);
                } else {
                    *it = c.n_ids[i] - r_start;it++;
                    *d_it = c.n_dists[i];d_it++;
                }
            }
            c.n_num -= c.remote_n_num;
            std::copy(remote_n_ids.begin(),remote_n_ids.end(), it);
            memcpy(c.n_ids, tmp.data(), (c.n_num + c.remote_n_num)*sizeof(int));
            std::copy(remote_n_dists.begin(),remote_n_dists.end(), d_it);
            memcpy(c.n_dists, dists.data(),
                   (c.n_num + c.remote_n_num)*sizeof(double));
        }
        std::sort(non_local_ids.begin(), non_local_ids.end());
        auto it = std::unique(non_local_ids.begin(), non_local_ids.end());
        non_local_ids.resize( std::distance(non_local_ids.begin(),it) );
        COLL_PRINT(work_comm, "non_local_ids.size(): %d",(int)non_local_ids.size());

        non_local_values.resize(non_local_ids.size());



        std::vector<int> r_starts(size), r_lens(size);
        MPI_Allgather(&r_start,1,MPI_INT,r_starts.data(),1,MPI_INT,work_comm);
        MPI_Allgather(&r_len,1,MPI_INT,r_lens.data(),1,MPI_INT,work_comm);


        std::map<int, int> aux;
        for (int i=0; i<r_starts.size(); i++) {
            aux[r_starts[i]]=i;
        }
        aux[node_number] = size;

        for (auto &n : non_local_ids) {
            int n_rank = aux.upper_bound(n)->second - 1;
            assert(n >= r_starts[n_rank] &&
                   n < r_starts[n_rank] + r_lens[n_rank]);

            auto it = non_loc_dependecies.find(n_rank);
            if (it == non_loc_dependecies.end()) {
                std::vector<int> ids;
                ids.push_back(n - r_starts[n_rank]);
                non_loc_dependecies[n_rank] = ids;
            } else {
                it->second.push_back(n - r_starts[n_rank]);

            }
        }



        for (auto &c : cnodes) {
            for (int i=c.n_num; i<c.n_num + c.remote_n_num; i++) {
                int n_id = c.n_ids[i];
                int n_rank = aux.upper_bound(n_id)->second - 1;
                int new_id = (int)(std::find(non_local_ids.begin(),
                                             non_local_ids.end(),
                                             n_id) -
                                   non_local_ids.begin());
                assert(new_id >= 0 && new_id < non_local_ids.size());
                c.n_ids[i] = new_id;
            }
        }
        auto it_nli = non_local_ids.begin();
        for (auto &d : non_loc_dependecies) {
            it_nli = std::copy(d.second.begin(),d.second.end(),it_nli);
        }

        std::stringstream iss;
        std::vector<int> num_remote_dependees(size);
        std::vector<int> num_local_dependencies(size);

        for (auto &n : non_loc_dependecies) {
            num_local_dependencies[n.first] = n.second.size();
            iss << "(" << n.first << ":" << n.second.size() <<") ";
        }

        COLL_PRINT(work_comm, "non_loc_dependecies: %s",iss.str().c_str());
        MPI_Alltoall(num_local_dependencies.data(),1,MPI_INT,
                     num_remote_dependees.data(),1,MPI_INT,
                     work_comm);
        iss.str(std::string());
        iss.clear();
        for (int i=0; i<num_remote_dependees.size(); i++)
            iss << "[" <<i<<":"<<num_remote_dependees[i]<<"] ";
        COLL_PRINT(work_comm, "num_remote_dependees: %s",iss.str().c_str());

        for (int i=0; i<num_remote_dependees.size(); i++) {
            if (num_remote_dependees[i] > 0) {
                std::vector<int> ids(num_remote_dependees[i]);
                non_loc_dependees[i] = ids;
            }
        }

        std::vector<MPI_Request> reqs(non_loc_dependecies.size()+
                                      non_loc_dependees.size());

        int count = 0;
        for (auto &n : non_loc_dependecies) {
            int remote_rank = n.first;
            MPI_Isend(n.second.data(),n.second.size(),MPI_INT,
                      remote_rank,123, work_comm,
                      &reqs[count++]);
        }
        for (auto &n : non_loc_dependees) {
            int remote_rank = n.first;
            MPI_Irecv(n.second.data(),n.second.size(),MPI_INT,
                      remote_rank, 123, work_comm,
                      &reqs[count++]);
        }
        assert(count == reqs.size());
        std::vector<MPI_Status> statuses(reqs.size());

        MPI_Waitall(reqs.size(), reqs.data(), statuses.data());

        for (auto &n : non_loc_dependees) {
            std::vector<double> buf(n.second.size());
            non_loc_dependees_bufs[n.first] = buf;
        }
        sreqs.resize(non_loc_dependees.size());
        rreqs.resize(non_loc_dependecies.size());
        sstats.resize(non_loc_dependees.size());
        rstats.resize(non_loc_dependecies.size());

    }

    void write_state(std::string filename) {
        int local_node_num = cnodes.size();
        int local_size = local_node_num*DynamicalSystem::SYS_SIZE;
        double *sbuf = new double[local_size];
        for (int i=0; i<local_node_num; i++) {
            memcpy(((char*)sbuf)+sizeof(double)*DynamicalSystem::SYS_SIZE*i,
                   cnodes[i].cell.Y, sizeof(double)*DynamicalSystem::SYS_SIZE);
        }
        if (work_rank == 0) {
            int *sizes = new int[work_size];
            MPI_Gather(&local_size,1,MPI_INT,
                       sizes, 1, MPI_INT, 0, work_comm);

            int nnum = 0;
            for (int i=0; i<work_size; i++)
                nnum+=sizes[i];
            assert(nnum == node_number*DynamicalSystem::SYS_SIZE);
            double *buf = new double[nnum];
            int *displs = new int[work_size];
            displs[0] = 0;
            for (int i=1; i<work_size; i++) {
                displs[i] = displs[i-1]+sizes[i-1];
            }

            for (int i=0; i<work_size; i++) {
                fprintf(stderr,"(%d:%d) ",sizes[i], displs[i]);
            }
            fprintf(stderr,"\n");
            MPI_Gatherv(sbuf,local_size,MPI_DOUBLE,
                        buf, sizes, displs, MPI_DOUBLE, 0, work_comm);

            std::ofstream fs(filename.c_str(), std::ios::out | std::ios::binary);
            fs.write(reinterpret_cast<const char*>(&work_size), sizeof(work_size));
            fs.write(reinterpret_cast<const char*>(sizes), sizeof(*sizes)*work_size);
            fs.write(reinterpret_cast<const char*>(displs), sizeof(*displs)*work_size);
            fs.write(reinterpret_cast<const char*>(buf), sizeof(*buf)*nnum);
            fs.write(reinterpret_cast<const char*>(&start_time), sizeof(start_time));
            fs.close();
            delete[] displs;
            delete[] sizes;
            delete[] buf;
        } else {
            MPI_Gather(&local_size,1,MPI_INT,
                       NULL, 0, MPI_INT, 0, work_comm);
            MPI_Gatherv(sbuf,local_size,MPI_DOUBLE,
                        NULL, NULL, NULL, MPI_DOUBLE, 0, work_comm);
        }

        delete[] sbuf;
    }

    void read_state(std::string filename) {
        int local_node_num = cnodes.size();
        int local_size = local_node_num*DynamicalSystem::SYS_SIZE;
        double *rbuf = new double[local_size];
        if (work_rank == 0){
            std::ifstream fs(filename.c_str(), std::ios::binary);
            int read_work_size;
            fs.read(reinterpret_cast<char*>(&read_work_size), sizeof(read_work_size));
            if (read_work_size != work_size) {
                fprintf(stderr, "Can't read input state file %s: work_size of the state file is %d"
                        " while current work_size is%d\n",filename.c_str(),
                        read_work_size, work_size);
                MPI_Abort(work_comm, -1);
            }

            int *sizes = new int[work_size];
            int *displs = new int[work_size];
            double *buf = new double[node_number*DynamicalSystem::SYS_SIZE];
            fs.read(reinterpret_cast<char*>(sizes), sizeof(*sizes)*work_size);
            fs.read(reinterpret_cast<char*>(displs), sizeof(*displs)*work_size);
            fs.read(reinterpret_cast<char*>(buf),
                    sizeof(*buf)*node_number*DynamicalSystem::SYS_SIZE);
            fs.read(reinterpret_cast<char*>(&start_time),
                    sizeof(start_time));

            for (int i=0; i<work_size; i++) {
                fprintf(stderr,"(%d:%d) ",sizes[i], displs[i]);
            }
            fprintf(stderr,"\n");

            MPI_Scatterv(buf, sizes, displs, MPI_DOUBLE,
                         rbuf, local_size, MPI_DOUBLE,0,work_comm);
            delete[] sizes;
            delete[] displs;
            delete[] buf;
        } else {
            MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE,
                         rbuf, local_size, MPI_DOUBLE,0,work_comm);

        }
        for (int i=0; i<local_node_num; i++) {
            memcpy(cnodes[i].cell.Y,
                   ((char*)rbuf)+sizeof(double)*DynamicalSystem::SYS_SIZE*i,
                   sizeof(double)*DynamicalSystem::SYS_SIZE);
        }
        MPI_Bcast(&start_time,1,MPI_INT,0,work_comm);

        delete[] rbuf;
    }
};


static inline void
usage(){
    std::cout << "Usage:"
        "\t ./heart_demo -m <mesh_file> [ -p <paced_cells_file> ] [-v]\n"
        "\t-m, --mesh_file  -  defines a VTK file containing tetrahedral  mesh\n"
        "\t-s, --setups_file  -  defines a file containing basic model params\n"
        "\t-v, --visualize  -  enable real time visualization\n"
        "\t-i, --improve-locality  -  do cuthill-mkcee reordering on the vtk mesh before calculations\n"
        "\t-b, --block-gather  -  wait for completion on each gatherv operation (only meaningfull with -v)\n"
        "\t-w, --write-images  -  save vizualization frames as images"<<
        std::endl;
}
        static inline int
parse_options(int argc, char **argv) {
    options.need_visualize = false;
    options.improve_locality = false;
    options.block_gather = false;
    options.save_images = false;
    options.solve_time     = 5.0;
    while (1)
    {
        static struct option long_options[] =
            {
                {"mesh-file",         required_argument,   0, 'm'},
                {"setups-file",       required_argument,   0, 's'},
                {"solve-time",        required_argument,   0, 't'},
                {"visualize",         no_argument,         0, 'v'},
                {"improve-locality",  no_argument,         0, 'i'},
                {"block-gather",      no_argument,         0, 'b'},
                {"save-images",       no_argument,         0, 'w'},
                {"read-state",      no_argument,           0, 'R'},
                {"write-state",      no_argument,          0, 'W'},
                {0, 0, 0, 0}
            };
        /* getopt_long stores the option index here. */
        int option_index = 0;
        int c;
        std::stringstream iss;

        c = getopt_long (argc, argv, "m:s:t:vibwW:R:",
                         long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c)
        {
        case 'm':
            options.mesh_file = std::string(optarg);
            break;
        case 'R':
            options.read_state = std::string(optarg);
            break;

        case 'W':
            options.write_state = std::string(optarg);
            break;

        case 's':
            options.setups_file = std::string(optarg);
            break;

        case 'v':
            options.need_visualize = true;
            break;
        case 'b':
            options.block_gather = true;
            break;
        case 'w':
            options.save_images = true;
            break;
        case 't':
            iss << std::string(optarg);
            iss >> options.solve_time;
            break;
        case 'i':
            options.improve_locality = true;
            break;
        default:
            usage();
            return -1;
        }
    }
    return 0;
}

int main ( int argc, char *argv[] )
{

    // setenv("LIBGL_ALWAYS_INDIRECT","y",1);



    if (parse_options(argc, argv))
        return -1;

    if (options.mesh_file.size() == 0) {
        usage();
        return -1;
    }

    // Mesh _mesh(options.mesh_file, Mesh::READ_MODE_BIN);
    // _mesh.saveMeshBinary("unn.bin");
    // return 0;
    MPI_Init(&argc, &argv);
//    MPI_Pcontrol(0,"");
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (options.need_visualize && size < 2) {
        fprintf(stderr,"At least 2 ranks are needed for visualization & computation mode\n");
        MPI_Finalize();
        exit(-1);
    }
    char hostname[100];
    gethostname(hostname,sizeof(hostname));
    fprintf(stderr,"MPI rank %d has started on %s\n",
            rank, hostname);

    MPI_Comm work_comm;
    if (options.need_visualize) {

        if (rank == size - 1) {

#if defined (ENABLE_GL)
            MPI_Group world_group, work_group;
            int rank_to_excl = size - 1;
            MPI_Comm_group(MPI_COMM_WORLD, &world_group);
            MPI_Group_excl(world_group, 1, &rank_to_excl, &work_group);
            MPI_Comm_create(MPI_COMM_WORLD, work_group, &work_comm);
            MPI_Group_free(&work_group);
            QApplication app(argc, argv);
            auto mesh = new Mesh(options.mesh_file, Mesh::READ_MODE_TXT);
//            MPI_Pcontrol(1,"");
            Window window(mesh);
            window.resize(window.sizeHint());
            int desktopArea = QApplication::desktop()->width() *
                QApplication::desktop()->height();
            int widgetArea = window.width() * window.height();

            window.SetSaveImages(options.save_images);
            window.SetDataRange(DATA_RANGE_MAX,
                                DATA_RANGE_MIN);
            window.show();
            window.resize(800,800);
            app.exec();
            MPI_Finalize();
            return 0;
#endif

        } else {
            MPI_Group world_group, work_group;
            int rank_to_excl = size - 1;


            MPI_Comm_group(MPI_COMM_WORLD, &world_group);
            MPI_Group_excl(world_group, 1, &rank_to_excl, &work_group);
            MPI_Comm_create(MPI_COMM_WORLD, work_group, &work_comm);
            MPI_Group_free(&work_group);
        }
    } else {
        work_comm = MPI_COMM_WORLD;
    }

    Mesh *mesh = NULL;
    Task task;
    task.setVisualize(options.need_visualize);

    int work_rank;
    MPI_Comm_rank(work_comm, &work_rank);
    if (work_rank == 0) {
        double t1 = omp_get_wtime();
        mesh = new Mesh(options.mesh_file,Mesh::READ_MODE_TXT);
        mesh->find_neighbours();
        mesh->calc_fem_matrices();
        task = Task(mesh, options.need_visualize);
        std::cout << "Root init took " <<
            omp_get_wtime()-t1 << " secs" << std::endl;
    }
    MPI_Barrier(work_comm);
//    MPI_Pcontrol(1,"");

    task.setVizRank(size-1);
    // task.setD(1e-4);
    // task.setD(2);
    // D = 2 for FHN

    task.setWorkComm(work_comm);
    task.distribute();
    if (options.read_state.size()){
        ROOT_PRINT(work_comm, "Reading state from %s", options.read_state.c_str());
        task.read_state(options.read_state);
    }

    ROOT_PRINT(work_comm, "Task initialized & distributed\n"
               "Solve time: %g ms with solve step %g\n"
               "starting solution...",options.solve_time,task.get_dt());
    task.solve(options.solve_time);


    ROOT_PRINT(work_comm, "Done");
    if (options.write_state.size()) {
        ROOT_PRINT(work_comm, "Writing state to %s", options.write_state.c_str());
        task.write_state(options.write_state);
    }
    if (work_comm != MPI_COMM_WORLD)
        MPI_Comm_free(&work_comm);
    MPI_Finalize();
    if (mesh)
        delete mesh;

    return EXIT_SUCCESS;
}
