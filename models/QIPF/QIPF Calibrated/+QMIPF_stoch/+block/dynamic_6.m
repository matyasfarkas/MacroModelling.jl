function [y, T, residual, g1] = dynamic_6(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(215)-params(118))/params(118))-(params(89)*(y(23)-params(118))/params(118)+1/params(72)*x(3));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(118);
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
