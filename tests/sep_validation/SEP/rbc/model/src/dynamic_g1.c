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

void dynamic_g1(const double *restrict y, const double *restrict x, const double *restrict params, const double *restrict steady_state, const double *restrict T, double *restrict g1_v)
{
g1_v[0]=(-(y[11]*T[13]*T[14]));
g1_v[1]=(-(1-params[5]));
g1_v[2]=(-params[7]);
g1_v[3]=1;
g1_v[4]=(-(T[11]*params[3]*(-y[15])/(y[7]*y[7])*T[15]));
g1_v[5]=1;
g1_v[6]=(-1);
g1_v[7]=(-((1-params[3])*1/y[9]*T[16]));
g1_v[8]=(-1);
g1_v[9]=(-(y[11]*T[14]*T[17]));
g1_v[10]=y[10]*(1-params[1])/params[1]/((1-y[9])*(1-y[9]))-(1-params[3])*T[16]*(-y[8])/(y[9]*y[9]);
g1_v[11]=T[19]*T[20]/y[10];
g1_v[12]=1;
g1_v[13]=(1-params[1])/params[1]/(1-y[9]);
g1_v[14]=(y[10]*T[20]*T[4]*T[25]-T[6])/(y[10]*y[10]);
g1_v[15]=1;
g1_v[16]=1;
g1_v[17]=(-T[2]);
g1_v[18]=1;
g1_v[19]=(-T[0]);
g1_v[20]=1;
g1_v[21]=(-(T[11]*params[3]*T[15]*1/y[7]));
g1_v[22]=(-(T[12]*T[24]));
g1_v[23]=(-(T[12]*T[27]));
g1_v[24]=(-params[8]);
}

