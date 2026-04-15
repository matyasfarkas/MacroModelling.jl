function [y, T, residual, g1] = static_7(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(91)-params(197))/params(197))-((y(91)-params(197))*params(70)/params(197)+x(21)+params(81)*x(22));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(197)-params(70)/params(197);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
