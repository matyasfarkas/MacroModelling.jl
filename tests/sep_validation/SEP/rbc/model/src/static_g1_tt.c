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

void static_g1_tt(const double *restrict y, const double *restrict x, const double *restrict params, double *restrict T)
{
T[8] = getPowerDeriv(T[1],1/params[4],1);
T[9] = getPowerDeriv(y[1]/y[0],1-params[4],1);
T[10] = getPowerDeriv(y[1]/y[2],1-params[4],1);
T[11] = getPowerDeriv(T[5],1-params[2],1);
T[12] = T[3]*(-getPowerDeriv(1-y[2],1-params[1],1))*T[11];
T[13] = T[11]*T[4]*getPowerDeriv(y[3],params[1],1);
}

