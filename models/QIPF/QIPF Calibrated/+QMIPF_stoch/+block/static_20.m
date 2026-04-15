function [y, T, residual, g1] = static_20(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(7)=log((1+params(83))/y(117));
  residual(1)=(T(7))-(x(19)+T(7)*params(67));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(83)))/(y(117)*y(117))/((1+params(83))/y(117))-params(67)*(-(1+params(83)))/(y(117)*y(117))/((1+params(83))/y(117));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
