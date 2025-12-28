#include <math.h>
#include "mex.h"

// The k-th derivative of x^p
inline double
getPowerDeriv(double x, double p, int k)
{
  if (fabs(x) < 1e-12 && p >= 0 && k > p && fabs(p-nearbyint(p)) < 1e-12)
    return 0.0;
  else
    {
      double dxp = pow(x, p-k);
      for (int i = 0; i<k; i++)
        dxp *= p--;
      return dxp;
    }
}

void dynamic_g2_tt(const double *restrict y, const double *restrict x, const double *restrict params, const double *restrict steady_state, double *restrict T)
{
T[28] = getPowerDeriv(T[1],1/params[4],2);
T[29] = getPowerDeriv(y[8]/y[9],1-params[4],2);
T[30] = getPowerDeriv(y[15]/y[7],1-params[4],2);
T[31] = getPowerDeriv(T[5],1-params[2],2);
T[32] = getPowerDeriv(T[9],1-params[2],2);
}

