function [y, T, residual, g1] = static_10(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(4)=log((1+params(83))/y(116));
  residual(1)=(T(4))-(T(4)*params(66)+x(18)+params(79)*x(19));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(83)))/(y(116)*y(116))/((1+params(83))/y(116))-params(66)*(-(1+params(83)))/(y(116)*y(116))/((1+params(83))/y(116));
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
