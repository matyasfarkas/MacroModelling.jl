function [y, T, residual, g1] = dynamic_18(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(log((1+params(82))/y(307)))-(params(65)*log((1+params(82))/y(115))+x(17));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(82)))/(y(307)*y(307))/((1+params(82))/y(307));
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
