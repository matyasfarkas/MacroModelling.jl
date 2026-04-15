function [y, T, residual, g1] = dynamic_20(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(log((1+params(83))/y(193)))-(x(18)+params(67)*log((1+params(83))/y(72)));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(83)))/(y(193)*y(193))/((1+params(83))/y(193));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
