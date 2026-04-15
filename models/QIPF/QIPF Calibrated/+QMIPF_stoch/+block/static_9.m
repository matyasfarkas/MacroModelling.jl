function [y, T, residual, g1] = static_9(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  T(3)=log((1+params(82))/y(113));
  residual(1)=(T(3))-(T(3)*params(63)+x(15));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(82)))/(y(113)*y(113))/((1+params(82))/y(113))-params(63)*(-(1+params(82)))/(y(113)*y(113))/((1+params(82))/y(113));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
