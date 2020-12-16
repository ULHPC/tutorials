#ifndef __MESH_HPP__
#define __MESH_HPP__
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <string.h>
using namespace std;


struct Point{
    float c[3];
    int id;
    vector<int> neighbours;
    vector<double> mass;
    vector<double> stiff;
    Point( int _id, double *_c) {
        id = _id;
        for (int i=0; i<3; i++)
            c[i] = (float)_c[i];
    }
    Point(){};
};

struct Tet{
    int points[4];
    int id;
    Tet(int _id, int *_points){
        id = _id;
        vector<int> p(_points, _points + 4);
        sort(p.begin(), p.end());
        for (int i=0; i<4; i++)
            points[i] = p[i];
    }
    Tet(){};
};

struct Facet{
    int points[3];
    int tet;
    int num_tets;
    float normal[3];
    Facet(){};
    Facet(int p1, int p2, int p3, int tet_id) {
        vector<int> pvec(3);
        pvec[0] = p1; pvec[1] = p2; pvec[2] = p3;
        sort(pvec.begin(), pvec.end());
        for (int i=0; i<3; i++)
            points[i] = pvec[i];
        tet = tet_id;
        num_tets = 1;
    }
    bool operator==(Facet &f) {
        return (points[0] == f.points[0]) &&
            (points[1] == f.points[1]) && (points[2] == f.points[2]);
    }
    bool operator!=(Facet &f) {
        return !(operator==(f));
    }

};

class Mesh {
    vector<Point> points;
    vector<Tet>   tets  ;
    vector<Facet> boundary_facets;
    void read_points(string filename);
    void read_tets(string filename);
    float xrange[2];
    float yrange[2];
    float zrange[2];
public:
    enum{
        READ_MODE_TXT,
        READ_MODE_BIN
    };
    Mesh(string filename, int read_mode);
    void build_facets();


    void read_mesh_binary(string filename);
    vector<Point>& getPoints() { return points; }
    vector<Facet>& getBoundaryFacets() { return boundary_facets; }
    vector<Tet>&   getTets() { return tets; }
    void GetPoint(int id, float *c) {
        memcpy(c,points[id].c,3*sizeof(float));
    }
    void normalizeForDrawing();
    void saveMeshBinary(string filename);
    void saveBoundaryFacets(string filename);
    void calcBoundaryFacetsNormals();
    float* GetPointC(int id) { return points[id].c; }
    void find_neighbours();
    void calc_fem_matrices();
    float getXmin() {return xrange[0];}
    float getXmax() {return xrange[1];}
    float getYmin() {return yrange[0];}
    float getYmax() {return yrange[1];}
    float getZmin() {return zrange[0];}
    float getZmax() {return zrange[1];}
    float getXrange() {return xrange[1]-xrange[0];}
    float getYrange() {return yrange[1]-yrange[0];}
    float getZrange() {return zrange[1]-zrange[0];}
    float getXcenter() {return (xrange[0]+xrange[1])/2.0;}
    float getYcenter() {return (yrange[0]+yrange[1])/2.0;}
    float getZcenter() {return (zrange[0]+zrange[1])/2.0;}

};
#endif
