function [y, T, residual, g1] = dynamic_8(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(127))-(params(7)*y(77)*y(193)*y(118));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(params(7)*y(77)*y(193)));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
