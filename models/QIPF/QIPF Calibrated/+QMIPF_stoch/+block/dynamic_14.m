function [y, T, residual, g1] = dynamic_14(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(295)-params(168))-(params(58)*(y(103)-params(168))+x(10));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1;
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
