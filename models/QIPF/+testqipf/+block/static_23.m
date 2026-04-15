function [y, T, residual, g1] = static_23(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(3)-params(107))/params(191))-((y(3)-params(107))*params(91)/params(191)+x(1));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(191)-params(91)/params(191);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
