function [y, T, residual, g1] = static_33(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(109))-(y(118)*(log(y(12)-y(12)*params(87)-params(114)*y(71))-T(91)*params(8)*y(124)/(1+params(7)))+params(3)*y(109));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(3);
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
