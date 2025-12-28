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

void static_resid_tt(const double *restrict y, const double *restrict x, const double *restrict params, double *restrict T)
{
T[0] = params[6]*exp(y[5]);
T[1] = params[3]*pow(y[0],params[4])+(1-params[3])*pow(y[2],params[4]);
T[2] = pow(T[1],1/params[4]);
T[3] = pow(y[3],params[1]);
T[4] = pow(1-y[2],1-params[1]);
T[5] = T[3]*T[4];
T[6] = pow(T[5],1-params[2]);
T[7] = 1+params[3]*pow(y[1]/y[0],1-params[4])-params[5];
}

