function [y, T, residual, g1] = static_32(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(110))-(y(119)*(log(y(14)-y(14)*params(88)-params(115)*y(72))-T(49)*params(9)*y(125)/(1+params(7)))+params(4)*y(110));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(4);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
