function [y, T, residual, g1] = dynamic_9(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(log((1+params(82))/y(305)))-(params(63)*log((1+params(82))/y(113))+x(15));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(82)))/(y(305)*y(305))/((1+params(82))/y(305));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
