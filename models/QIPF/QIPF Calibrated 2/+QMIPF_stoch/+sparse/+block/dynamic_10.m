function [y, T, residual, g1] = dynamic_10(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=(log((1+params(83))/y(308)))-(params(66)*log((1+params(83))/y(116))+x(18)+params(79)*x(19));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=(-(1+params(83)))/(y(308)*y(308))/((1+params(83))/y(308));
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
