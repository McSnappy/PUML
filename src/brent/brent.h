//
// https://people.sc.fsu.edu/~jburkardt/c_src/brent/brent.html
//
// modified 1/9/17 to allow void *user in local_min()
//
double glomin ( double a, double b, double c, double m, double machep, 
  double e, double t, double f ( double x ), double *x );
double local_min ( double a, double b, double eps, double t, 
		   double f ( double x, void *user ), void *user, double *x );
double local_min_rc ( double *a, double *b, int *status, double value );
double r8_epsilon ( );
double r8_max ( double x, double y );
double r8_sign ( double x );
void timestamp ( );
double zero ( double a, double b, double machep, double t, 
  double f ( double x ) );
void zero_rc ( double a, double b, double t, double *arg, int *status, 
  double value );
