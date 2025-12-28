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

void static_2_resid(double *restrict y, const double *restrict x, const double *restrict params, double *restrict T)
{
  y[4]=params[6]*exp(y[5]);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (nrhs != 7)
    mexErrMsgTxt("Accepts exactly 7 input arguments");
  if (nlhs != 2)
    mexErrMsgTxt("Accepts exactly 2 output arguments");
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
  if (!(mxIsDouble(prhs[6]) && !mxIsComplex(prhs[6]) && !mxIsSparse(prhs[6]) && mxGetNumberOfElements(prhs[6]) >= 0))
    mexErrMsgTxt("T must be a real dense numeric array with at least 0 elements");
  plhs[1] = mxDuplicateArray(prhs[6]);
  double *restrict T = mxGetDoubles(plhs[1]);
  static_2_resid(y, x, params, T);
}
