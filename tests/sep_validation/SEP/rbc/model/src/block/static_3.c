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

void static_3_resid(double *restrict y, const double *restrict x, const double *restrict params, double *restrict T, double *restrict residual)
{
  T[0]=params[3]*pow(y[0],params[4])+(1-params[3])*pow(y[2],params[4]);
  residual[0]=(y[1])-(y[4]*pow(T[0],1/params[4]));
  residual[1]=(y[0])-(y[1]-y[3]+y[0]*(1-params[5]));
  residual[2]=(y[3]*(1-params[1])/params[1]/(1-y[2])-(1-params[3])*pow(y[1]/y[2],1-params[4]))-(0);
  T[1]=pow(y[3],params[1]);
  T[2]=pow(1-y[2],1-params[1]);
  T[3]=T[1]*T[2];
  T[4]=pow(T[3],1-params[2]);
  T[5]=1+params[3]*pow(y[1]/y[0],1-params[4])-params[5];
  residual[3]=(T[4]/y[3])-(T[4]*params[0]/y[3]*T[5]);
  T[6]=getPowerDeriv(T[0],1/params[4],1);
  T[7]=getPowerDeriv(y[1]/y[0],1-params[4],1);
  T[8]=getPowerDeriv(y[1]/y[2],1-params[4],1);
  T[9]=getPowerDeriv(T[3],1-params[2],1);
  T[10]=T[1]*(-getPowerDeriv(1-y[2],1-params[1],1))*T[9];
  T[11]=T[9]*T[2]*getPowerDeriv(y[3],params[1],1);
}

void static_3_g1(const double *restrict y, const double *restrict x, const double *restrict params, double *restrict T, double *restrict g1_v)
{
g1_v[0]=(-(y[4]*params[3]*getPowerDeriv(y[0],params[4],1)*T[6]));
g1_v[1]=1-(1-params[5]);
g1_v[2]=(-(T[4]*params[0]/y[3]*params[3]*(-y[1])/(y[0]*y[0])*T[7]));
g1_v[3]=1;
g1_v[4]=(1-params[1])/params[1]/(1-y[2]);
g1_v[5]=(y[3]*T[11]-T[4])/(y[3]*y[3])-T[5]*(y[3]*params[0]*T[11]-T[4]*params[0])/(y[3]*y[3]);
g1_v[6]=(-(y[4]*T[6]*(1-params[3])*getPowerDeriv(y[2],params[4],1)));
g1_v[7]=y[3]*(1-params[1])/params[1]/((1-y[2])*(1-y[2]))-(1-params[3])*T[8]*(-y[1])/(y[2]*y[2]);
g1_v[8]=T[10]/y[3]-T[5]*params[0]*T[10]/y[3];
g1_v[9]=1;
g1_v[10]=(-1);
g1_v[11]=(-((1-params[3])*1/y[2]*T[8]));
g1_v[12]=(-(T[4]*params[0]/y[3]*params[3]*T[7]*1/y[0]));
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (nrhs != 7)
    mexErrMsgTxt("Accepts exactly 7 input arguments");
  if (nlhs < 2 || nlhs > 4)
    mexErrMsgTxt("Accepts 2 to 4 output arguments");
  if (!(mxIsDouble(prhs[0]) && !mxIsComplex(prhs[0]) && !mxIsSparse(prhs[0]) && mxGetNumberOfElements(prhs[0]) == 7))
    mexErrMsgTxt("y must be a real dense numeric array with 7 elements");
  if (!(mxIsDouble(prhs[1]) && !mxIsComplex(prhs[1]) && !mxIsSparse(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1))
    mexErrMsgTxt("x must be a real dense numeric array with 1 elements");
  const double *restrict x = mxGetDoubles(prhs[1]);
  if (!(mxIsDouble(prhs[2]) && !mxIsComplex(prhs[2]) && !mxIsSparse(prhs[2]) && mxGetNumberOfElements(prhs[2]) == 9))
    mexErrMsgTxt("params must be a real dense numeric array with 9 elements");
  const double *restrict params = mxGetDoubles(prhs[2]);
  plhs[0] = mxDuplicateArray(prhs[0]);
  double *restrict y = mxGetDoubles(plhs[0]);
  if (!(mxIsInt32(prhs[3]) && mxGetNumberOfElements(prhs[3]) == 13))
    mexErrMsgTxt("sparse_rowval must be an int32 array with 13 elements");
  if (!(mxIsInt32(prhs[5]) && mxGetNumberOfElements(prhs[5]) == 5))
    mexErrMsgTxt("sparse_colptr must be an int32 array with 5 elements");
  const int32_T *restrict sparse_rowval = mxGetInt32s(prhs[3]);
  const int32_T *restrict sparse_colptr = mxGetInt32s(prhs[5]);
  if (!(mxIsDouble(prhs[6]) && !mxIsComplex(prhs[6]) && !mxIsSparse(prhs[6]) && mxGetNumberOfElements(prhs[6]) >= 12))
    mexErrMsgTxt("T must be a real dense numeric array with at least 12 elements");
  plhs[1] = mxDuplicateArray(prhs[6]);
  double *restrict T = mxGetDoubles(plhs[1]);
  mxArray *residual_mx = mxCreateDoubleMatrix(4, 1, mxREAL);
  double *restrict residual = mxGetDoubles(residual_mx);
  static_3_resid(y, x, params, T, residual);
  if (nlhs > 2)
    plhs[2] = residual_mx;
  else
    mxDestroyArray(residual_mx);
  if (nlhs > 3)
    {
  plhs[3] = mxCreateSparse(4, 4, 13, mxREAL);
  mwIndex *restrict ir = mxGetIr(plhs[3]), *restrict jc = mxGetJc(plhs[3]);
  for (mwSize i = 0; i < 13; i++)
    *ir++ = *sparse_rowval++ - 1;
  for (mwSize i = 0; i < 5; i++)
    *jc++ = *sparse_colptr++ - 1;
      double *restrict g1_v = mxGetDoubles(plhs[3]);
      static_3_g1(y, x, params, T, g1_v);
    }
}
