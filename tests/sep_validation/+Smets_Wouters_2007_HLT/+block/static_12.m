function [y, T] = static_12(y, x, params, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(51)=100*(y(48)-1);
  y(37)=params(18)+100*(y(35)/(y(35))-1);
  y(43)=100*(y(42)-1);
  y(64)=100*log(y(62)/y(63));
end
