#include "mesh.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <assert.h>
#include <math.h>
#include <stdio.h>

template class vector<Facet>;
void Mesh::read_points(string filename) {
    xrange[0] = 1e15; xrange[1] = -1e15;
    yrange[0] = 1e15; yrange[1] = -1e15;
    zrange[0] = 1e15; zrange[1] = -1e15;
    int points_number;
    ifstream ifs;
    ifs.exceptions ( ifstream::failbit | ifstream::badbit );
    try{
        ifs.open(filename.c_str(), ios::in | ios::binary);
    }
    catch(ifstream::failure e){
        cout << "Error: Can not open input mesh file " << filename << ": " << e.what() << endl;
    }
    ifs >> points_number;
    points.resize(points_number);
    for (int i=0; i<points_number; i++) {
        ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        int id;
        double c[3];
        ifs >> id >> c[0] >> c[1] >> c[2];
        points[id] = Point(id,c);
        if (c[0] < xrange[0]) xrange[0] = c[0];
        if (c[0] > xrange[1]) xrange[1] = c[0];

        if (c[1] < yrange[0]) yrange[0] = c[1];
        if (c[1] > yrange[1]) yrange[1] = c[1];

        if (c[2] < zrange[0]) zrange[0] = c[2];
        if (c[2] > zrange[1]) zrange[1] = c[2];
        
    }
    ifs.close();

}

void Mesh::read_tets(string filename) {
    int tets_number;
    ifstream ifs;
    ifs.exceptions ( ifstream::failbit | ifstream::badbit );
    try{
        ifs.open(filename.c_str(), ios::in | ios::binary);
    }
    catch(ifstream::failure e){
        cout << "Error: Can not open input mesh file " << filename << ": " << e.what() << endl;
    }
    ifs >> tets_number;
    tets.resize(tets_number);
    for (int i=0; i<tets_number; i++) {
        ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        int id;
        int v[4];
        ifs >> id >> v[0] >> v[1] >> v[2] >> v[3];
        tets[id] = Tet(id,v);
    }
    ifs.close();
}


void Mesh::build_facets() {
    int combinations[4][3] = {{0,1,2},{0,1,3}, {0,2,3}, {1,2,3}};
    vector<Facet> facets;
    facets.resize(tets.size()*4);
    int j = 0;
    for (auto &t : tets) {
        for (int i=0; i<4; i++) {
            int *c = combinations[i];
            facets[j++] = Facet(t.points[c[0]], t.points[c[1]],
                                t.points[c[2]], t.id);
        }
    }
    stable_sort(facets.begin(), facets.end(),
                [=](const Facet &f1, const Facet &f2)->bool {
                    if (f1.points[0] != f2.points[0]) return f1.points[0] < f2.points[0];
                    else if (f1.points[1] != f2.points[1]) return f1.points[1] < f2.points[1];
                    else return f1.points[2] < f2.points[2];

                    // return (f1.points[0] < f2.points[0]) ||
                    // (f1.points[1] < f2.points[1]) ||
                    // (f1.points[2] < f2.points[2]);
                }
        );
    auto it = facets.begin();
    for (auto cit = it + 1; cit != facets.end(); cit++) {
        if (*it != *cit) {
            it = cit;
        } else {
            it->num_tets++;
        }
    }


    auto uit = unique(facets.begin(), facets.end());
    facets.erase(uit,facets.end());

    for (auto &f : facets)
        if (f.num_tets == 1) {
            boundary_facets.push_back(f);
        }
    cout << "ALL facets: " << facets.size()
         << " BFacets: " << boundary_facets.size() << endl;
}

Mesh::Mesh(string filename, int read_mode) {

    if (read_mode == Mesh::READ_MODE_TXT) {
        read_points(filename+".node");
        read_tets(filename+".ele");
    } else {
        read_mesh_binary(filename);
    }

}

void Mesh::normalizeForDrawing() {
    float center[3] = {0, 0, 0};
    float c_max[3] = {-1e10,-1e10,-1e10};
    float c_min[3]  = {1e10, 1e10, 1e10};
    for (auto &p : points) {
        float *c = p.c;
        for (int i=0; i<3; i++) {
            center[i]+=c[i];
            if (c[i] > c_max[i]) c_max[i] = c[i];
            if (c[i] < c_min[i]) c_min[i] = c[i];
        }
    }
    float width[3];
    for (int i=0; i<3; i++){
        center[i] /= points.size();
        width[i] = (c_max[i] - c_min[i])/2.0;
    }

    for (auto &p : points) {
        float *c = p.c;
        for (int i=0; i<3; i++) {
            c[i] = (c[i] - center[i])/width[i];
        }
    }
}

void Mesh::calcBoundaryFacetsNormals() {
    for (auto &f : boundary_facets){
        float *p1 = GetPointC(f.points[0]);
        float *p2 = GetPointC(f.points[1]);
        float *p3 = GetPointC(f.points[2]);

        float v1[3] = {p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]};
        float v2[3] = {p3[0]-p1[0],p3[1]-p1[1],p3[2]-p1[2]};
        float *normal = f.normal;
        normal[0] = v1[1]*v2[2]-v1[2]*v2[1];
        normal[1] = v1[2]*v2[0]-v1[0]*v2[2];
        normal[2] = v1[0]*v2[1]-v1[1]*v2[0];

        int* tet_points = tets[f.tet].points;
        int extra;
        for (int i=0; i<4; i++) {
            if ((tet_points[i] != f.points[0]) &&
                (tet_points[i] != f.points[1]) &&
                (tet_points[i] != f.points[2])) {
                extra = tet_points[i];
                break;
            }
        }

        float* p_extra = points[extra].c;
        float v_check[3] = {p_extra[0]-p1[0],p_extra[1]-p1[1],p_extra[2]-p1[2]};
        float v = 0;
        for (int i=0; i<3; i++)
            v += v_check[i]*normal[i];
        if (v>0) {
            for (int i=0; i<3; i++)
                normal[i] = -normal[i];
            int p = f.points[2];
            f.points[2] = f.points[1];
            f.points[1] = p;
        }
    }
}

void Mesh::saveMeshBinary(string filename) {
    // FILE *ofs = fopen(filename.c_str(), "w");
    // fprintf(ofs,"%d %d\n", points.size(), tets.size());
    // ofs = freopen(filename.c_str(), "ab", ofs);
    // for (int i=0; i<points.size(); i++) {
        // int id = points[i].getId();
        // fwrite(&id,1,sizeof(int),ofs);
        // fwrite(points[i].getC(),3,sizeof(double),ofs);
    // }
    // for (int i=0; i<tets.size(); i++) {
        // auto &t = tets[i];
        // Point** pts = t.getPoints();
        // int buf[5] = {t.getId(), pts[0]->getId(), pts[1]->getId(), pts[2]->getId(), pts[3]->getId()};
        // fwrite(buf,5,sizeof(int),ofs);
    // }
    // fclose(ofs);
}

void Mesh::read_mesh_binary(string filename) {
    FILE *ifs = fopen(filename.c_str(), "r");
    // int npoints, ntets;
    // fscanf(ifs,"%d %d\n", &npoints, &ntets);
    // cout << npoints << " " << ntets << endl;
    // ifs = freopen(filename.c_str(), "rb", ifs);
    // points.reserve(npoints);
    // for (int i=0; i<npoints; i++) {
        // int id;
        // double c[3];
        // fread(&id,1,sizeof(int),ifs);
        // fread(c,3,sizeof(double),ifs);
        // points.push_back(Point(id,c[0],c[1],c[2]));

    // }
    // tets.reserve(ntets);
    // for (int i=0; i<ntets; i++) {
        // int buf[5];
        // fread(buf,5,sizeof(int),ifs);
        // tets.push_back(Tet(buf[0],
                           // &points[buf[1]],
                           // &points[buf[2]],
                           // &points[buf[3]],
                           // &points[buf[4]]));
    // }
    fclose(ifs);

}

void Mesh::saveBoundaryFacets(string filename) {
    // FILE *ofs = fopen(filename.c_str(), "w");
    // fprintf(ofs,"%d\n", boundary_facets.size());
    // ofs = freopen(filename.c_str(), "ab", ofs);
    // for (int i=0; i<boundary_facets.size(); i++) {
        // int id = points[i].getId();
        // fwrite(&id,1,sizeof(int),ofs);
        // fwrite(points[i].getC(),3,sizeof(double),ofs);
    // }
    // fclose(ofs);
}

void Mesh::find_neighbours() {
    int cell_num = tets.size();
    for (auto i=0; i<cell_num; i++) {
        int* tet_points = tets[i].points;
        for (auto j=0; j<4; j++) {
            int _p = tet_points[j];
            assert(points[_p].id == _p);
            for (auto k=0; k<4; k++) {
                int p = tet_points[k];
                points[_p].neighbours.push_back(p);
            }
        }
    }

    int point_num = points.size();
    for (auto i=0; i<point_num; i++) {
        auto &n_ids = points[i].neighbours;
        std::sort(n_ids.begin(), n_ids.end());
        auto it = std::unique(n_ids.begin(), n_ids.end());
        n_ids.resize( std::distance(n_ids.begin(),it) );
        points[i].mass.resize(n_ids.size());
        points[i].stiff.resize(n_ids.size());
    }
}


static double m1[4][4] =
{
    {2/120.0, 1/120.0, 1/120.0, 1/120.0},
    {1/120.0, 2/120.0, 1/120.0, 1/120.0},
    {1/120.0, 1/120.0, 2/120.0, 1/120.0},
    {1/120.0, 1/120.0, 1/120.0, 2/120.0}
};



static inline double
calc_det(const double m[][3]) {
    return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-
        m[0][1]*(m[1][0]*m[2][2]-m[2][0]*m[1][2])+
        m[0][2]*(m[1][0]*m[2][1]-m[2][0]*m[1][1]);
}

static inline void
init_bmat(double Bmat[][3], const float *c1, const float *c2,
          const float *c3, const float *c4) {
    Bmat[0][0] = c2[0]-c1[0]; Bmat[0][1] = c3[0]-c1[0]; Bmat[0][2] = c4[0]-c1[0];
    Bmat[1][0] = c2[1]-c1[1]; Bmat[1][1] = c3[1]-c1[1]; Bmat[1][2] = c4[1]-c1[1];
    Bmat[2][0] = c2[2]-c1[2]; Bmat[2][1] = c3[2]-c1[2]; Bmat[2][2] = c4[2]-c1[2];

}

static inline void
transpose_invert(double in[][3], double out[][3], double invdet) {
    out[0][0] =  (in[1][1]*in[2][2]-in[2][1]*in[1][2])*invdet;
    out[1][0] = -(in[0][1]*in[2][2]-in[0][2]*in[2][1])*invdet;
    out[2][0] =  (in[0][1]*in[1][2]-in[0][2]*in[1][1])*invdet;
    out[0][1] = -(in[1][0]*in[2][2]-in[1][2]*in[2][0])*invdet;
    out[1][1] =  (in[0][0]*in[2][2]-in[0][2]*in[2][0])*invdet;
    out[2][1] = -(in[0][0]*in[1][2]-in[1][0]*in[0][2])*invdet;
    out[0][2] =  (in[1][0]*in[2][1]-in[2][0]*in[1][1])*invdet;
    out[1][2] = -(in[0][0]*in[2][1]-in[2][0]*in[0][1])*invdet;
    out[2][2] =  (in[0][0]*in[1][1]-in[1][0]*in[0][1])*invdet;
}

static inline void
transpose(double in[][3], double out[][3]) {
    out[0][0] = in[0][0]; out[0][1] = in[1][0]; out[0][2] = in[2][0];
    out[1][0] = in[0][1]; out[1][1] = in[1][1]; out[1][2] = in[2][1];
    out[2][0] = in[0][2]; out[2][1] = in[1][2]; out[2][2] = in[2][2];
}

static inline void
mult_3x3(double m1[][3], double m2[][3], double rst[][3])
{
    int i, j, k;
    for(i = 0; i < 3; i++)
    {
        rst[i][0] = rst[i][1] = rst[i][2] = 0.0;
        for(j = 0; j < 3; j++)
        {
            for(k = 0; k < 3; k++)
            {
                rst[i][j] +=  m1[i][k] *  m2[k][j];
            }
        }
    }
}

static double K_xi_xi[4][4]=
{
    { 1, -1, 0, 0},
    {-1,  1, 0, 0},
    { 0,  0, 0, 0},
    { 0,  0, 0, 0}
};

static double K_eta_eta[4][4]=
{
    { 1, 0, -1, 0},
    { 0, 0,  0, 0},
    {-1, 0,  1, 0},
    { 0, 0,  0, 0}
};

static double K_nu_nu[4][4]=
{
    { 1, 0, 0, -1},
    { 0, 0, 0,  0},
    { 0, 0, 0,  0},
    {-1, 0, 0,  1}
};

static double K_eta_xi[4][4]=
{
    { 1, -1, 0, 0},
    { 0,  0, 0, 0},
    {-1,  1, 0, 0},
    { 0,  0, 0, 0}
};

static double K_nu_xi[4][4]=
{
    { 1, -1, 0, 0},
    { 0,  0, 0, 0},
    { 0,  0, 0, 0},
    {-1,  1, 0, 0}
};

static double K_nu_eta[4][4]=
{
    { 1, 0, -1,  0},
    { 0, 0,  0,  0},
    { 0, 0,  0,  0},
    {-1, 0,  1,  0}
};


void Mesh::calc_fem_matrices() {
    int cell_num = tets.size();
    int i, j, k;
    double Bmat[3][3];
    double Bmat_ti[3][3];
    double Bmat_i[3][3];
    double C[3][3];
    for (i=0; i<cell_num; i++) {

        int* tet_points = tets[i].points;
        init_bmat(Bmat,points[tet_points[0]].c,
                  points[tet_points[1]].c,
                  points[tet_points[2]].c,
                  points[tet_points[3]].c);

        double detB = calc_det(Bmat);

        transpose_invert(Bmat, Bmat_ti, 1.0/detB);
        detB = fabs(detB);
        transpose(Bmat_ti, Bmat_i);
        mult_3x3(Bmat_i, Bmat_ti, C);
        for (j=0; j<4; j++) {
            int _p = tet_points[j];
            auto &P = points[_p];
            for (k=0; k<4; k++) {
                int p = tet_points[k];
                int ii=0;
                for (; ii<P.neighbours.size(); ii++)
                    if (p == P.neighbours[ii]) break;
                P.mass[ii] += m1[j][k]*detB;
                P.stiff[ii] += detB/6.0*(C[0][0]*K_xi_xi[j][k]+C[1][1]*K_eta_eta[j][k]+C[2][2]*K_nu_nu[j][k]+
                                         C[0][1]*(K_eta_xi[j][k]+K_eta_xi[k][j]) +
                                         C[0][2]*(K_nu_xi[j][k]+K_nu_xi[k][j]) +
                                         C[1][2]*(K_nu_eta[j][k]+K_nu_eta[k][j]));
            }
        }
    }
}
