function [y, T, residual, g1] = static_13(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(61))-(params(7)*y(11)*y(61)*y(52));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(params(7)*y(11)*y(61)));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
