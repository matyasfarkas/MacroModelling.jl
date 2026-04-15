function [y, T, residual, g1] = dynamic_17(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(369)-params(228))/params(228))-(x(23)+params(71)*(y(177)-params(228))/params(228));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(228);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
