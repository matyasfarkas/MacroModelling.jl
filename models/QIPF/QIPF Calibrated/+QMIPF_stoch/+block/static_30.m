function [y, T, residual, g1] = static_30(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(124))-((1-params(97))*(y(132)/y(126))^T(47)+T(92)*params(97)*y(124));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(97)*T(92);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
