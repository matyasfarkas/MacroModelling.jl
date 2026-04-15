function [y, T, residual, g1] = dynamic_17(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(241)-params(226))/params(226))-(x(22)+params(71)*(y(120)-params(226))/params(226));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(226);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
