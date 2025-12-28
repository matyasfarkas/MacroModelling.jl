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

void dynamic_g1_tt(const double *restrict y, const double *restrict x, const double *restrict params, const double *restrict steady_state, double *restrict T)
{
T[13] = params[3]*getPowerDeriv(y[0],params[4],1);
T[14] = getPowerDeriv(T[1],1/params[4],1);
T[15] = getPowerDeriv(y[15]/y[7],1-params[4],1);
T[16] = getPowerDeriv(y[8]/y[9],1-params[4],1);
T[17] = (1-params[3])*getPowerDeriv(y[9],params[4],1);
T[18] = (-getPowerDeriv(1-y[9],1-params[1],1));
T[19] = T[3]*T[18];
T[20] = getPowerDeriv(T[5],1-params[2],1);
T[21] = (-getPowerDeriv(1-y[16],1-params[1],1));
T[22] = T[7]*T[21];
T[23] = getPowerDeriv(T[9],1-params[2],1);
T[24] = params[0]*T[22]*T[23]/y[17];
T[25] = getPowerDeriv(y[10],params[1],1);
T[26] = getPowerDeriv(y[17],params[1],1);
T[27] = (y[17]*params[0]*T[23]*T[8]*T[26]-T[10])/(y[17]*y[17]);
}

