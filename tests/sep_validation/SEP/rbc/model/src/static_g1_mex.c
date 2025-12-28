#include <string.h>
#include "mex.h"
#include "static_g1.h"
#include "static_resid_tt.h"
#include "static_g1_tt.h"

#define max(a, b) ((a > b) ? (a) : (b))

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (nrhs != 6 && nrhs != 8)
    mexErrMsgTxt("Accepts exactly 6 or 8 input arguments");
  if (nlhs != 1 && nlhs != 3)
    mexErrMsgTxt("Accepts exactly 1 or 3 output arguments");
  if (!(mxIsDouble(prhs[0]) && !mxIsComplex(prhs[0]) && !mxIsSparse(prhs[0]) && mxGetNumberOfElements(prhs[0]) == 7))
    mexErrMsgTxt("y must be a real dense numeric array with 7 elements");
  const double *restrict y = mxGetDoubles(prhs[0]);
  if (!(mxIsDouble(prhs[1]) && !mxIsComplex(prhs[1]) && !mxIsSparse(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1))
    mexErrMsgTxt("x must be a real dense numeric array with 1 elements");
  const double *restrict x = mxGetDoubles(prhs[1]);
  if (!(mxIsDouble(prhs[2]) && !mxIsComplex(prhs[2]) && !mxIsSparse(prhs[2]) && mxGetNumberOfElements(prhs[2]) == 9))
    mexErrMsgTxt("params must be a real dense numeric array with 9 elements");
  const double *restrict params = mxGetDoubles(prhs[2]);
  if (!(mxIsInt32(prhs[3]) && mxGetNumberOfElements(prhs[3]) == 20))
    mexErrMsgTxt("sparse_rowval must be an int32 array with 20 elements");
  if (!(mxIsInt32(prhs[5]) && mxGetNumberOfElements(prhs[5]) == 8))
    mexErrMsgTxt("sparse_colptr must be an int32 array with 8 elements");
  const int32_T *restrict sparse_rowval = mxGetInt32s(prhs[3]);
  const int32_T *restrict sparse_colptr = mxGetInt32s(prhs[5]);
  mxArray *T_mx, *T_order_mx;
  int T_order_on_input;
  if (nrhs > 6)
    {
      T_order_mx = (mxArray *) prhs[6];
      T_mx = (mxArray *) prhs[7];
      if (!(mxIsScalar(T_order_mx) && mxIsNumeric(T_order_mx)))
        mexErrMsgTxt("T_order should be a numeric scalar");
      if (!(mxIsDouble(T_mx) && !mxIsComplex(T_mx) && !mxIsSparse(T_mx) && mxGetN(T_mx) == 1))
        mexErrMsgTxt("T_mx should be a real dense column vector");
      T_order_on_input = mxGetScalar(T_order_mx);
      if (T_order_on_input < 1)
        {
          T_order_mx = mxCreateDoubleScalar(1);
          const mxArray *T_old_mx = T_mx;
          T_mx = mxCreateDoubleMatrix(max(14, mxGetM(T_old_mx)), 1, mxREAL);
          memcpy(mxGetDoubles(T_mx), mxGetDoubles(T_old_mx), mxGetM(T_old_mx)*sizeof(double));
        }
      else if (mxGetM(T_mx) < 14)
        mexErrMsgTxt("T_mx should have at least 14 elements");
    }
  else
    {
      T_order_mx = mxCreateDoubleScalar(1);
      T_mx = mxCreateDoubleMatrix(14, 1, mxREAL);
      T_order_on_input = -1;
    }
  double *restrict T = mxGetDoubles(T_mx);
  if (T_order_on_input < 1)
    switch (T_order_on_input)
      {
      default:
        static_resid_tt(y, x, params, T);
      case 0:
        static_g1_tt(y, x, params, T);
      }
  plhs[0] = mxCreateSparse(7, 7, 20, mxREAL);
  mwIndex *restrict ir = mxGetIr(plhs[0]), *restrict jc = mxGetJc(plhs[0]);
  for (mwSize i = 0; i < 20; i++)
    *ir++ = *sparse_rowval++ - 1;
  for (mwSize i = 0; i < 8; i++)
    *jc++ = *sparse_colptr++ - 1;
  static_g1(y, x, params, T, mxGetDoubles(plhs[0]));
  if (nlhs == 3)
    {
      plhs[1] = T_order_mx;
      plhs[2] = T_mx;
    }
}
