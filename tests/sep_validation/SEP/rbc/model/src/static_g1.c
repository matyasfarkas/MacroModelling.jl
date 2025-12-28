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

void static_g1(const double *restrict y, const double *restrict x, const double *restrict params, const double *restrict T, double *restrict g1_v)
{
g1_v[0]=(-(y[4]*params[3]*getPowerDeriv(y[0],params[4],1)*T[8]));
g1_v[1]=1-(1-params[5]);
g1_v[2]=(-(T[6]*params[0]/y[3]*params[3]*(-y[1])/(y[0]*y[0])*T[9]));
g1_v[3]=1;
g1_v[4]=(-1);
g1_v[5]=(-((1-params[3])*1/y[2]*T[10]));
g1_v[6]=(-(T[6]*params[0]/y[3]*params[3]*T[9]*1/y[0]));
g1_v[7]=(-1);
g1_v[8]=(-(y[4]*T[8]*(1-params[3])*getPowerDeriv(y[2],params[4],1)));
g1_v[9]=y[3]*(1-params[1])/params[1]/((1-y[2])*(1-y[2]))-(1-params[3])*T[10]*(-y[1])/(y[2]*y[2]);
g1_v[10]=T[12]/y[3]-T[7]*params[0]*T[12]/y[3];
g1_v[11]=1;
g1_v[12]=(1-params[1])/params[1]/(1-y[2]);
g1_v[13]=(y[3]*T[13]-T[6])/(y[3]*y[3])-T[7]*(y[3]*params[0]*T[13]-T[6]*params[0])/(y[3]*y[3]);
g1_v[14]=1;
g1_v[15]=1;
g1_v[16]=(-T[2]);
g1_v[17]=1-params[7];
g1_v[18]=(-T[0]);
g1_v[19]=1;
}

