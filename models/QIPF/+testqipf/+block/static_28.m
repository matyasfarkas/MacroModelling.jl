function [y, T, residual, g1] = static_28(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(80))-((1-params(98))*(y(84)/y(82))^T(20)+T(23)*params(98)*y(80));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(98)*T(23);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
