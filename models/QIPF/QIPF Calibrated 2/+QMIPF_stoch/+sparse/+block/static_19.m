function [y, T, residual, g1] = static_19(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(6)=log((1+params(82))/y(114));
  residual(1)=(T(6))-(T(6)*params(64)+x(16));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(82)))/(y(114)*y(114))/((1+params(82))/y(114))-params(64)*(-(1+params(82)))/(y(114)*y(114))/((1+params(82))/y(114));
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
