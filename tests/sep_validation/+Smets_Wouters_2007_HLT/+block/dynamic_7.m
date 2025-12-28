function [y, T] = dynamic_7(y, x, params, steady_state, sparse_rowval, sparse_colval, sparse_colptr, T)
  y(117)=100*(y(114)-1);
  y(109)=100*(y(108)-1);
  y(84)=params(37)+100*(y(121)/y(55)-1);
  y(103)=params(18)+100*(y(101)/(steady_state(35))-1);
  y(81)=params(37)+100*(y(95)/y(29)-1);
  y(80)=params(37)+100*(y(78)/y(12)-1);
  y(85)=params(37)+100*(y(128)/y(62)-1);
  y(130)=100*log(y(128)/y(129));
end
