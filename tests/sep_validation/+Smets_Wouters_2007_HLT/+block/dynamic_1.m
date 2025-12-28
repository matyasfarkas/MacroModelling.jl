function [y, T] = dynamic_1(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(72)=1-params(23)+params(23)*y(6)+params(43)/100*x(1);
  y(77)=1-params(24)+params(24)*y(11)+params(44)/100*params(1)*x(2);
end
