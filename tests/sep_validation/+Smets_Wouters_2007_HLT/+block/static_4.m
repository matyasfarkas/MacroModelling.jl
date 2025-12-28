function [y, T, residual, g1] = static_4(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(28)-params(9))-((y(28)-params(9))*params(25)+params(45)/100*x(3)+params(43)/100*x(1)*params(11));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(25);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
