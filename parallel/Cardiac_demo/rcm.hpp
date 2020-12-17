/* This source file is downloaded from
 *  http://people.sc.fsu.edu/~jburkardt/cpp_src/rcm/rcm.html
 *
 *   It is distributed under GNU LGPL license.
 * Modified: 23.04.2015
 */

int adj_bandwidth ( int node_num, int adj_num, int adj_row[], int adj[] );
bool adj_contains_ij ( int node_num, int adj_num, int adj_row[], int adj[], 
  int i, int j );
void adj_insert_ij ( int node_num, int adj_max, int *adj_num, int adj_row[], 
  int adj[], int i, int j );
int adj_perm_bandwidth ( int node_num, int adj_num, int adj_row[], int adj[], 
  int perm[], int perm_inv[] );
void adj_perm_show ( int node_num, int adj_num, int adj_row[], int adj[], 
  int perm[], int perm_inv[] );
void adj_print ( int node_num, int adj_num, int adj_row[], int adj[], 
  string title );
void adj_print_some ( int node_num, int node_lo, int node_hi, int adj_num, 
  int adj_row[], int adj[], string title );
void adj_set ( int node_num, int adj_max, int *adj_num, int adj_row[], 
  int adj[], int irow, int jcol );
void adj_show ( int node_num, int adj_num, int adj_row[], int adj[] );
void degree ( int root, int adj_num, int adj_row[], int adj[], int mask[], 
  int deg[], int *iccsze, int ls[], int node_num );
void genrcm ( int node_num, int adj_num, int adj_row[], int adj[], int perm[] );
void graph_01_adj ( int node_num, int adj_num, int adj_row[], int adj[] );
void graph_01_size ( int *node_num, int *adj_num );
int i4_max ( int i1, int i2 );
int i4_min ( int i1, int i2 );
int i4_sign ( int i );
void i4_swap ( int *i, int *j );
int i4_uniform ( int a, int b, int *seed );
int i4col_compare ( int m, int n, int a[], int i, int j );
void i4col_sort_a ( int m, int n, int a[] );
void i4col_swap ( int m, int n, int a[], int irow1, int irow2 );
void i4mat_print_some ( int m, int n, int a[], int ilo, int jlo, int ihi, 
  int jhi, string title );
void i4mat_transpose_print ( int m, int n, int a[], string title );
void i4mat_transpose_print_some ( int m, int n, int a[], int ilo, int jlo, 
  int ihi, int jhi, string title );
void i4vec_heap_d ( int n, int a[] );
int *i4vec_indicator ( int n );
void i4vec_print ( int n, int a[], string title );
void i4vec_reverse ( int n, int a[] );
void i4vec_sort_heap_a ( int n, int a[] );
void level_set ( int root, int adj_num, int adj_row[], int adj[], int mask[], 
  int *level_num, int level_row[], int level[], int node_num );
void level_set_print ( int node_num, int level_num, int level_row[], 
  int level[] );
bool perm_check ( int n, int p[] );
void perm_inverse3 ( int n, int perm[], int perm_inv[] );
int *perm_uniform ( int n, int *seed );
float r4_abs ( float x );
int r4_nint ( float x );
void r82vec_permute ( int n, double a[], int p[] );
void r8mat_print_some ( int m, int n, double a[], int ilo, int jlo, int ihi, 
  int jhi, string title );
void r8mat_transpose_print_some ( int m, int n, double a[], int ilo, int jlo, 
  int ihi, int jhi, string title );
void rcm ( int root, int adj_num, int adj_row[], int adj[], int mask[], 
  int perm[], int *iccsze, int node_num );
void root_find ( int *root, int adj_num, int adj_row[], int adj[], int mask[], 
  int *level_num, int level_row[], int level[], int node_num );
void sort_heap_external ( int n, int *indx, int *i, int *j, int isgn );
void timestamp ( );
int *triangulation_neighbor_triangles ( int triangle_order, int triangle_num,
  int triangle_node[] );
int triangulation_order3_adj_count ( int node_num, int triangle_num, 
  int triangle_node[], int triangle_neighbor[], int adj_col[] );
int *triangulation_order3_adj_set ( int node_num, int triangle_num,
  int triangle_node[], int triangle_neighbor[], int adj_num, int adj_col[] );
void triangulation_order3_example2 ( int node_num, int triangle_num, 
  double node_xy[], int triangle_node[], int triangle_neighbor[] );
void triangulation_order3_example2_size ( int *node_num, int *triangle_num,
  int *hole_num );
int triangulation_order6_adj_count ( int node_num, int triangle_num, 
  int triangle_node[], int triangle_neighbor[], int adj_col[] );
int *triangulation_order6_adj_set ( int node_num, int triangle_num, 
  int triangle_node[], int triangle_neighbor[], int adj_num, int adj_col[] );
void triangulation_order6_example2 ( int node_num, int triangle_num, 
  double node_xy[], int triangle_node[], int triangle_neighbor[] );
void triangulation_order6_example2_size ( int *node_num, int *triangle_num,
  int *hole_num );
