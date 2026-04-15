function [y, T, residual, g1] = static_8(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(2)=log((1+params(82))/y(112));
  residual(1)=(T(2))-(T(2)*params(62)+x(14));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(82)))/(y(112)*y(112))/((1+params(82))/y(112))-params(62)*(-(1+params(82)))/(y(112)*y(112))/((1+params(82))/y(112));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
