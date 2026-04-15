function [y, T, residual, g1] = static_28(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(125))-((1-params(98))*(y(133)/y(128))^T(47)+T(50)*params(98)*y(125));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(98)*T(50);
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
