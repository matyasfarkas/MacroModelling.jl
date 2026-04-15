function [y, T, residual, g1] = dynamic_3(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(y(263)-params(144))-(params(55)*(y(71)-params(144))+x(7));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1;
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
