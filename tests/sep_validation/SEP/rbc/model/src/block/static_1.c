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

void static_1_resid(double *restrict y, const double *restrict x, const double *restrict params, double *restrict T, double *restrict residual)
{
  residual[0]=(y[5])-(y[5]*params[7]+params[8]*x[0]);
}

void static_1_g1(const double *restrict y, const double *restrict x, const double *restrict params, double *restrict T, double *restrict g1_v)
{
g1_v[0]=1-params[7];
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
  if (!(mxIsInt32(prhs[3]) && mxGetNumberOfElements(prhs[3]) == 1))
    mexErrMsgTxt("sparse_rowval must be an int32 array with 1 elements");
  if (!(mxIsInt32(prhs[5]) && mxGetNumberOfElements(prhs[5]) == 2))
    mexErrMsgTxt("sparse_colptr must be an int32 array with 2 elements");
  const int32_T *restrict sparse_rowval = mxGetInt32s(prhs[3]);
  const int32_T *restrict sparse_colptr = mxGetInt32s(prhs[5]);
  if (!(mxIsDouble(prhs[6]) && !mxIsComplex(prhs[6]) && !mxIsSparse(prhs[6]) && mxGetNumberOfElements(prhs[6]) >= 0))
    mexErrMsgTxt("T must be a real dense numeric array with at least 0 elements");
  plhs[1] = mxDuplicateArray(prhs[6]);
  double *restrict T = mxGetDoubles(plhs[1]);
  mxArray *residual_mx = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *restrict residual = mxGetDoubles(residual_mx);
  static_1_resid(y, x, params, T, residual);
  if (nlhs > 2)
    plhs[2] = residual_mx;
  else
    mxDestroyArray(residual_mx);
  if (nlhs > 3)
    {
  plhs[3] = mxCreateSparse(1, 1, 1, mxREAL);
  mwIndex *restrict ir = mxGetIr(plhs[3]), *restrict jc = mxGetJc(plhs[3]);
  for (mwSize i = 0; i < 1; i++)
    *ir++ = *sparse_rowval++ - 1;
  for (mwSize i = 0; i < 2; i++)
    *jc++ = *sparse_colptr++ - 1;
      double *restrict g1_v = mxGetDoubles(plhs[3]);
      static_1_g1(y, x, params, T, g1_v);
    }
}
