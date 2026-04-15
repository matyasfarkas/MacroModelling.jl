function [y, T, residual, g1] = static_19(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(5)=log((1+params(82))/y(69));
  residual(1)=(T(5))-(T(5)*params(64)+x(15));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(82)))/(y(69)*y(69))/((1+params(82))/y(69))-params(64)*(-(1+params(82)))/(y(69)*y(69))/((1+params(82))/y(69));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
