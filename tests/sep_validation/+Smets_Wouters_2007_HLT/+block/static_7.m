function [y, T] = static_7(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(20)=params(47)/100*params(2)*x(5);
  y(21)=params(49)/100*params(4)*x(7);
  y(5)=0;
  y(4)=0;
  y(19)=params(37);
  y(14)=params(37);
  y(15)=params(37);
  y(18)=params(37);
  y(47)=y(46);
end
