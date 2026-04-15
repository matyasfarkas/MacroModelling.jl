function [y, T, residual, g1] = static_14(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(62)-params(166))-(x(9)+(y(62)-params(166))*params(58));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(58);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
