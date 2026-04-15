function [y, T, residual, g1] = dynamic_5(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(297)-params(170))-(params(60)*(y(105)-params(170))+x(12));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1;
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
