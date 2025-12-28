#include <string.h>
#include "mex.h"
#include "dynamic_g1.h"
#include "dynamic_resid_tt.h"
#include "dynamic_g1_tt.h"

#define max(a, b) ((a > b) ? (a) : (b))

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (nrhs != 7 && nrhs != 9)
    mexErrMsgTxt("Accepts exactly 7 or 9 input arguments");
  if (nlhs != 1 && nlhs != 3)
    mexErrMsgTxt("Accepts exactly 1 or 3 output arguments");
  if (!(mxIsDouble(prhs[0]) && !mxIsComplex(prhs[0]) && !mxIsSparse(prhs[0]) && mxGetNumberOfElements(prhs[0]) == 21))
    mexErrMsgTxt("y must be a real dense numeric array with 21 elements");
  const double *restrict y = mxGetDoubles(prhs[0]);
  if (!(mxIsDouble(prhs[1]) && !mxIsComplex(prhs[1]) && !mxIsSparse(prhs[1]) && mxGetNumberOfElements(prhs[1]) == 1))
    mexErrMsgTxt("x must be a real dense numeric array with 1 elements");
  const double *restrict x = mxGetDoubles(prhs[1]);
  if (!(mxIsDouble(prhs[2]) && !mxIsComplex(prhs[2]) && !mxIsSparse(prhs[2]) && mxGetNumberOfElements(prhs[2]) == 9))
    mexErrMsgTxt("params must be a real dense numeric array with 9 elements");
  const double *restrict params = mxGetDoubles(prhs[2]);
  if (!(mxIsDouble(prhs[3]) && !mxIsComplex(prhs[3]) && !mxIsSparse(prhs[3]) && mxGetNumberOfElements(prhs[3]) == 7))
    mexErrMsgTxt("steady_state must be a real dense numeric array with 7 elements");
  const double *restrict steady_state = mxGetDoubles(prhs[3]);
  if (!(mxIsInt32(prhs[4]) && mxGetNumberOfElements(prhs[4]) == 25))
    mexErrMsgTxt("sparse_rowval must be an int32 array with 25 elements");
  if (!(mxIsInt32(prhs[6]) && mxGetNumberOfElements(prhs[6]) == 23))
    mexErrMsgTxt("sparse_colptr must be an int32 array with 23 elements");
  const int32_T *restrict sparse_rowval = mxGetInt32s(prhs[4]);
  const int32_T *restrict sparse_colptr = mxGetInt32s(prhs[6]);
  mxArray *T_mx, *T_order_mx;
  int T_order_on_input;
  if (nrhs > 7)
    {
      T_order_mx = (mxArray *) prhs[7];
      T_mx = (mxArray *) prhs[8];
      if (!(mxIsScalar(T_order_mx) && mxIsNumeric(T_order_mx)))
        mexErrMsgTxt("T_order should be a numeric scalar");
      if (!(mxIsDouble(T_mx) && !mxIsComplex(T_mx) && !mxIsSparse(T_mx) && mxGetN(T_mx) == 1))
        mexErrMsgTxt("T_mx should be a real dense column vector");
      T_order_on_input = mxGetScalar(T_order_mx);
      if (T_order_on_input < 1)
        {
          T_order_mx = mxCreateDoubleScalar(1);
          const mxArray *T_old_mx = T_mx;
          T_mx = mxCreateDoubleMatrix(max(28, mxGetM(T_old_mx)), 1, mxREAL);
          memcpy(mxGetDoubles(T_mx), mxGetDoubles(T_old_mx), mxGetM(T_old_mx)*sizeof(double));
        }
      else if (mxGetM(T_mx) < 28)
        mexErrMsgTxt("T_mx should have at least 28 elements");
    }
  else
    {
      T_order_mx = mxCreateDoubleScalar(1);
      T_mx = mxCreateDoubleMatrix(28, 1, mxREAL);
      T_order_on_input = -1;
    }
  double *restrict T = mxGetDoubles(T_mx);
  if (T_order_on_input < 1)
    switch (T_order_on_input)
      {
      default:
        dynamic_resid_tt(y, x, params, steady_state, T);
      case 0:
        dynamic_g1_tt(y, x, params, steady_state, T);
      }
  plhs[0] = mxCreateSparse(7, 22, 25, mxREAL);
  mwIndex *restrict ir = mxGetIr(plhs[0]), *restrict jc = mxGetJc(plhs[0]);
  for (mwSize i = 0; i < 25; i++)
    *ir++ = *sparse_rowval++ - 1;
  for (mwSize i = 0; i < 23; i++)
    *jc++ = *sparse_colptr++ - 1;
  dynamic_g1(y, x, params, steady_state, T, mxGetDoubles(plhs[0]));
  if (nlhs == 3)
    {
      plhs[1] = T_order_mx;
      plhs[2] = T_mx;
    }
}
