function [y, T, residual, g1] = static_10(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(3)=log((1+params(83))/y(71));
  residual(1)=(T(3))-(T(3)*params(66)+x(17)+params(79)*x(18));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(83)))/(y(71)*y(71))/((1+params(83))/y(71))-params(66)*(-(1+params(83)))/(y(71)*y(71))/((1+params(83))/y(71));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
