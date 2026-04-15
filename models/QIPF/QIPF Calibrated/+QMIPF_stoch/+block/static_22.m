function [y, T, residual, g1] = static_22(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(5)-params(110))/params(193))-((y(5)-params(110))*params(92)/params(193)+x(2));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(193)-params(92)/params(193);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
