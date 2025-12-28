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

void dynamic_3_resid(double *restrict y, const double *restrict x, const double *restrict params, const double *restrict steady_state, double *restrict T)
{
  y[13]=y[8]-y[10];
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (nrhs != 8)
    mexErrMsgTxt("Accepts exactly 8 input arguments");
  if (nlhs != 2)
    mexErrMsgTxt("Accepts exactly 2 output arguments");
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
  if (!(mxIsDouble(prhs[7]) && !mxIsComplex(prhs[7]) && !mxIsSparse(prhs[7]) && mxGetNumberOfElements(prhs[7]) >= 17))
    mexErrMsgTxt("T must be a real dense numeric array with at least 17 elements");
  plhs[1] = mxDuplicateArray(prhs[7]);
  double *restrict T = mxGetDoubles(plhs[1]);
  dynamic_3_resid(y, x, params, steady_state, T);
}
