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

void dynamic_g2(const double *restrict y, const double *restrict x, const double *restrict params, const double *restrict steady_state, const double *restrict T, double *restrict g2_v)
{
g2_v[0]=(-T[0]);
g2_v[1]=(-(y[11]*(T[14]*params[3]*getPowerDeriv(y[0],params[4],2)+T[13]*T[13]*T[28])));
g2_v[2]=(-(y[11]*T[13]*T[17]*T[28]));
g2_v[3]=(-(T[13]*T[14]));
g2_v[4]=(-(y[11]*(T[17]*T[17]*T[28]+T[14]*(1-params[3])*getPowerDeriv(y[9],params[4],2))));
g2_v[5]=(-(T[14]*T[17]));
g2_v[6]=(-((1-params[3])*1/y[9]*1/y[9]*T[29]));
g2_v[7]=(-((1-params[3])*(T[16]*(-1)/(y[9]*y[9])+1/y[9]*(-y[8])/(y[9]*y[9])*T[29])));
g2_v[8]=(-(y[10]*(1-params[1])/params[1]*((-(1-y[9]))-(1-y[9]))))/((1-y[9])*(1-y[9])*(1-y[9])*(1-y[9]))-(1-params[3])*((-y[8])/(y[9]*y[9])*(-y[8])/(y[9]*y[9])*T[29]+T[16]*(-((-y[8])*(y[9]+y[9])))/(y[9]*y[9]*y[9]*y[9]));
g2_v[9]=(1-params[1])/params[1]/((1-y[9])*(1-y[9]));
g2_v[10]=(-(T[11]*params[3]*(T[15]*(-((-y[15])*(y[7]+y[7])))/(y[7]*y[7]*y[7]*y[7])+(-y[15])/(y[7]*y[7])*(-y[15])/(y[7]*y[7])*T[30])));
g2_v[11]=(-(T[11]*params[3]*(T[15]*(-1)/(y[7]*y[7])+(-y[15])/(y[7]*y[7])*1/y[7]*T[30])));
g2_v[12]=(-(params[3]*(-y[15])/(y[7]*y[7])*T[15]*T[24]));
g2_v[13]=(-(params[3]*(-y[15])/(y[7]*y[7])*T[15]*T[27]));
g2_v[14]=(-(T[11]*params[3]*1/y[7]*1/y[7]*T[30]));
g2_v[15]=(-(params[3]*T[15]*1/y[7]*T[24]));
g2_v[16]=(-(params[3]*T[15]*1/y[7]*T[27]));
g2_v[17]=(T[20]*T[3]*getPowerDeriv(1-y[9],1-params[1],2)+T[19]*T[19]*T[31])/y[10];
g2_v[18]=(y[10]*(T[20]*T[18]*T[25]+T[19]*T[4]*T[25]*T[31])-T[19]*T[20])/(y[10]*y[10]);
g2_v[19]=(-(T[12]*params[0]*(T[23]*T[7]*getPowerDeriv(1-y[16],1-params[1],2)+T[22]*T[22]*T[32])/y[17]));
g2_v[20]=(-(T[12]*(y[17]*params[0]*(T[23]*T[21]*T[26]+T[22]*T[8]*T[26]*T[32])-params[0]*T[22]*T[23])/(y[17]*y[17])));
g2_v[21]=(y[10]*y[10]*y[10]*(T[4]*T[25]*T[4]*T[25]*T[31]+T[20]*T[4]*getPowerDeriv(y[10],params[1],2))-(y[10]*T[20]*T[4]*T[25]-T[6])*(y[10]+y[10]))/(y[10]*y[10]*y[10]*y[10]);
g2_v[22]=(-(T[12]*(y[17]*y[17]*y[17]*params[0]*(T[8]*T[26]*T[8]*T[26]*T[32]+T[23]*T[8]*getPowerDeriv(y[17],params[1],2))-(y[17]*params[0]*T[23]*T[8]*T[26]-T[10])*(y[17]+y[17]))/(y[17]*y[17]*y[17]*y[17])));
}

