function [y, T, residual, g1] = static_23(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
residual=NaN(1, 1);
  residual(1)=((y(4)-params(109))/params(193))-((y(5)-params(110))*params(229)/params(193)-y(108)*params(230)+x(1));
if nargout > 3
    g1_v = NaN(1, 1);
g1_v(1)=1/params(193);
    if ~isoctave && matlab_ver_less_than('9.8')
        sparse_rowval = double(sparse_rowval);
        sparse_colval = double(sparse_colval);
    end
    g1 = sparse(sparse_rowval, sparse_colval, g1_v, 1, 1);
end
end
