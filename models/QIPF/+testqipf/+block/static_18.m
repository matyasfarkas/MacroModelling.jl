function [y, T, residual, g1] = static_18(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(4)=log((1+params(82))/y(70));
  residual(1)=(T(4))-(T(4)*params(65)+x(16));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(82)))/(y(70)*y(70))/((1+params(82))/y(70))-params(65)*(-(1+params(82)))/(y(70)*y(70))/((1+params(82))/y(70));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
