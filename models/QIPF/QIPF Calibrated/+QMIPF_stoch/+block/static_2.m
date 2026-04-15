function [y, T, residual, g1] = static_2(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(118)-params(181))-((y(118)-params(181))*params(68)+x(20)+params(80)*x(21));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(68);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
