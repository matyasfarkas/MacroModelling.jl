function [y, T, residual, g1] = dynamic_22(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(125)-params(108))/params(191))-(params(92)*(y(4)-params(108))/params(191)+x(2));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(191);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
