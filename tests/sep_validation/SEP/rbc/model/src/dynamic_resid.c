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
extern inline double getPowerDeriv(double x, double p, int k);

void dynamic_resid(const double *restrict y, const double *restrict x, const double *restrict params, const double *restrict steady_state, const double *restrict T, double *restrict residual)
{
    residual[0] = (y[12]) - (params[7]*y[5]+params[8]*x[0]);
    residual[1] = (y[11]) - (T[0]);
    residual[2] = (y[8]) - (y[11]*T[2]);
    residual[3] = (y[7]) - (y[8]-y[10]+y[0]*(1-params[5]));
residual[4] = y[10]*(1-params[1])/params[1]/(1-y[9])-(1-params[3])*pow(y[8]/y[9],1-params[4]);
    residual[5] = (T[6]/y[10]) - (T[11]*T[12]);
    residual[6] = (y[13]) - (y[8]-y[10]);
}

