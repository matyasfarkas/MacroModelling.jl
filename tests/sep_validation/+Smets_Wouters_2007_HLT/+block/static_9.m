function [y, T, residual, g1] = static_9(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(54))-(1-params(29)+y(54)*params(29)+y(21)-y(21)*params(17));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(29);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
