function [y, T, residual, g1] = dynamic_7(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(340)-params(199))/params(199))-(params(70)*(y(148)-params(199))/params(199)+x(22)+params(81)*x(23));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(199);
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
