function [y, T, residual, g1] = static_6(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(23)-params(118))/params(118))-((y(23)-params(118))*params(89)/params(118)+1/params(72)*x(3));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(118)-params(89)/params(118);
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
