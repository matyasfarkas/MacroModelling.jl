function [y, T, residual, g1] = static_26(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(79))-((1-params(97))*(y(83)/y(81))^T(20)+T(62)*params(97)*y(79));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(97)*T(62);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
