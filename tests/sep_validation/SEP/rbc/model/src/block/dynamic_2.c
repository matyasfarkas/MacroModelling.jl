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

void dynamic_2_resid(double *restrict y, const double *restrict x, const double *restrict params, const double *restrict steady_state, double *restrict T, double *restrict residual)
{
  T[1]=pow(y[10],params[1]);
  T[2]=pow(1-y[9],1-params[1]);
  T[3]=T[1]*T[2];
  T[4]=pow(T[3],1-params[2]);
  T[5]=pow(y[17],params[1]);
  T[6]=pow(1-y[16],1-params[1]);
  T[7]=T[5]*T[6];
  T[8]=params[0]*pow(T[7],1-params[2]);
  T[9]=T[8]/y[17];
  T[10]=1+params[3]*pow(y[15]/y[7],1-params[4])-params[5];
  residual[0]=(T[4]/y[10])-(T[9]*T[10]);
  T[11]=params[3]*pow(y[0],params[4])+(1-params[3])*pow(y[9],params[4]);
  residual[1]=(y[8])-(y[11]*pow(T[11],1/params[4]));
  residual[2]=(y[7])-(y[8]-y[10]+y[0]*(1-params[5]));
  residual[3]=(y[10]*(1-params[1])/params[1]/(1-y[9])-(1-params[3])*pow(y[8]/y[9],1-params[4]))-(0);
  T[12]=getPowerDeriv(T[11],1/params[4],1);
  T[13]=getPowerDeriv(y[15]/y[7],1-params[4],1);
  T[14]=getPowerDeriv(y[8]/y[9],1-params[4],1);
  T[15]=getPowerDeriv(T[3],1-params[2],1);
  T[16]=getPowerDeriv(T[7],1-params[2],1);
}

void dynamic_2_g1(const double *restrict y, const double *restrict x, const double *restrict params, const double *restrict steady_state, double *restrict T, double *restrict g1_v)
{
g1_v[0]=(-(y[11]*params[3]*getPowerDeriv(y[0],params[4],1)*T[12]));
g1_v[1]=(-(1-params[5]));
g1_v[2]=(-(T[9]*params[3]*(-y[15])/(y[7]*y[7])*T[13]));
g1_v[3]=1;
g1_v[4]=T[1]*(-getPowerDeriv(1-y[9],1-params[1],1))*T[15]/y[10];
g1_v[5]=(-(y[11]*T[12]*(1-params[3])*getPowerDeriv(y[9],params[4],1)));
g1_v[6]=y[10]*(1-params[1])/params[1]/((1-y[9])*(1-y[9]))-(1-params[3])*T[14]*(-y[8])/(y[9]*y[9]);
g1_v[7]=1;
g1_v[8]=(-1);
g1_v[9]=(-((1-params[3])*1/y[9]*T[14]));
g1_v[10]=(y[10]*T[15]*T[2]*getPowerDeriv(y[10],params[1],1)-T[4])/(y[10]*y[10]);
g1_v[11]=1;
g1_v[12]=(1-params[1])/params[1]/(1-y[9]);
g1_v[13]=(-(T[10]*params[0]*T[5]*(-getPowerDeriv(1-y[16],1-params[1],1))*T[16]/y[17]));
g1_v[14]=(-(T[9]*params[3]*T[13]*1/y[7]));
g1_v[15]=(-(T[10]*(y[17]*params[0]*T[16]*T[6]*getPowerDeriv(y[17],params[1],1)-T[8])/(y[17]*y[17])));
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (nrhs != 8)
    mexErrMsgTxt("Accepts exactly 8 input arguments");
  if (nlhs < 2 || nlhs > 4)
    mexErrMsgTxt("Accepts 2 to 4 output arguments");
  if (!(mxIsDouble(prhs[0]) && !mxIsComplex(prhs[0]) && !mxIsSparse(prhs[0]) && mxGetNumberOfElements(prhs[0]) == 21))
    mexErrMsgTxt("y must be a real dense numeric array with 21 elements");
  if (!(mxIsDouble(prhs[1]) && !mxIsComplex(prhs[1]) && !mxIsSparse(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1))
    mexErrMsgTxt("x must be a real dense numeric array with 1 elements");
  const double *restrict x = mxGetDoubles(prhs[1]);
  if (!(mxIsDouble(prhs[2]) && !mxIsComplex(prhs[2]) && !mxIsSparse(prhs[2]) && mxGetNumberOfElements(prhs[2]) == 9))
    mexErrMsgTxt("params must be a real dense numeric array with 9 elements");
  const double *restrict params = mxGetDoubles(prhs[2]);
  if (!(mxIsDouble(prhs[3]) && !mxIsComplex(prhs[3]) && !mxIsSparse(prhs[3]) && mxGetNumberOfElements(prhs[3]) == 7))
    mexErrMsgTxt("steady_state must be a real dense numeric array with 7 elements");
  const double *restrict steady_state = mxGetDoubles(prhs[3]);
  plhs[0] = mxDuplicateArray(prhs[0]);
  double *restrict y = mxGetDoubles(plhs[0]);
  if (!(mxIsInt32(prhs[4]) && mxGetNumberOfElements(prhs[4]) == 16))
    mexErrMsgTxt("sparse_rowval must be an int32 array with 16 elements");
  if (!(mxIsInt32(prhs[6]) && mxGetNumberOfElements(prhs[6]) == 13))
    mexErrMsgTxt("sparse_colptr must be an int32 array with 13 elements");
  const int32_T *restrict sparse_rowval = mxGetInt32s(prhs[4]);
  const int32_T *restrict sparse_colptr = mxGetInt32s(prhs[6]);
  if (!(mxIsDouble(prhs[7]) && !mxIsComplex(prhs[7]) && !mxIsSparse(prhs[7]) && mxGetNumberOfElements(prhs[7]) >= 17))
    mexErrMsgTxt("T must be a real dense numeric array with at least 17 elements");
  plhs[1] = mxDuplicateArray(prhs[7]);
  double *restrict T = mxGetDoubles(plhs[1]);
  mxArray *residual_mx = mxCreateDoubleMatrix(4, 1, mxREAL);
  double *restrict residual = mxGetDoubles(residual_mx);
  dynamic_2_resid(y, x, params, steady_state, T, residual);
  if (nlhs > 2)
    plhs[2] = residual_mx;
  else
    mxDestroyArray(residual_mx);
  if (nlhs > 3)
    {
  plhs[3] = mxCreateSparse(4, 12, 16, mxREAL);
  mwIndex *restrict ir = mxGetIr(plhs[3]), *restrict jc = mxGetJc(plhs[3]);
  for (mwSize i = 0; i < 16; i++)
    *ir++ = *sparse_rowval++ - 1;
  for (mwSize i = 0; i < 13; i++)
    *jc++ = *sparse_colptr++ - 1;
      double *restrict g1_v = mxGetDoubles(plhs[3]);
      dynamic_2_g1(y, x, params, steady_state, T, g1_v);
    }
}
