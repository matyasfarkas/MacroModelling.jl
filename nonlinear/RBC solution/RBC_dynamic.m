function [residual, g1, g2, g3] = RBC_dynamic(y, x, params, steady_state, it_)
%
% Status : Computes dynamic model for Dynare
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

%
% Model equations
%

residual = zeros(4, 1);
lhs =1/y(2);
rhs =params(2)/y(5)*(params(1)*exp(y(4))*y(3)^(params(1)-1)+1-params(5));
residual(1)= lhs-rhs;
lhs =y(7);
rhs =(-y(5))+y(3)*(1-params(5))+y(6);
residual(2)= lhs-rhs;
lhs =y(6);
rhs =exp(y(4))*y(3)^params(1);
residual(3)= lhs-rhs;
lhs =y(4);
rhs =params(4)*y(1)+x(it_, 1);
residual(4)= lhs-rhs;
if nargout >= 2,
  g1 = zeros(4, 8);

  %
  % Jacobian matrix
  %

  g1(1,3)=(-(params(2)/y(5)*params(1)*exp(y(4))*getPowerDeriv(y(3),params(1)-1,1)));
  g1(1,4)=(-(params(2)/y(5)*params(1)*exp(y(4))*y(3)^(params(1)-1)));
  g1(1,2)=(-1)/(y(2)*y(2));
  g1(1,5)=(-((params(1)*exp(y(4))*y(3)^(params(1)-1)+1-params(5))*(-params(2))/(y(5)*y(5))));
  g1(2,3)=(-(1-params(5)));
  g1(2,7)=1;
  g1(2,5)=1;
  g1(2,6)=(-1);
  g1(3,3)=(-(exp(y(4))*getPowerDeriv(y(3),params(1),1)));
  g1(3,4)=(-(exp(y(4))*y(3)^params(1)));
  g1(3,6)=1;
  g1(4,1)=(-params(4));
  g1(4,4)=1;
  g1(4,8)=(-1);
end
if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],4,64);
end
if nargout >= 4,
  %
  % Third order derivatives
  %

  g3 = sparse([],[],[],4,512);
end
end
