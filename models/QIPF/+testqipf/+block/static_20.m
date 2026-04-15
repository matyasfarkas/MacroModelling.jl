function [y, T, residual, g1] = static_20(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(6)=log((1+params(83))/y(72));
  residual(1)=(T(6))-(x(18)+T(6)*params(67));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(83)))/(y(72)*y(72))/((1+params(83))/y(72))-params(67)*(-(1+params(83)))/(y(72)*y(72))/((1+params(83))/y(72));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
