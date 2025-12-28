function [y, T] = dynamic_3(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(112)=1-params(28)+params(28)*y(46)+params(48)/100*params(3)*x(6);
  y(105)=1-params(26)+params(26)*y(39)+params(46)/100*x(4);
  y(86)=params(47)/100*params(2)*x(5);
  y(87)=params(49)/100*params(4)*x(7);
end
