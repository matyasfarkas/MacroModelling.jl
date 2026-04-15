function [y, T, residual, g1] = static_3(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(71)-params(144))-((y(71)-params(144))*params(55)+x(7));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1-params(55);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
