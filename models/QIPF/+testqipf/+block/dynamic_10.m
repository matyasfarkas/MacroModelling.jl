function [y, T, residual, g1] = dynamic_10(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(log((1+params(83))/y(192)))-(params(66)*log((1+params(83))/y(71))+x(17)+params(79)*x(18));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(83)))/(y(192)*y(192))/((1+params(83))/y(192));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
