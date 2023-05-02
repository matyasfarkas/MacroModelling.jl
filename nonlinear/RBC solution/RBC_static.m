function [residual, g1, g2] = RBC_static(y, x, params)
%
% Status : Computes static model for Dynare
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

residual = zeros( 4, 1);

%
% Model equations
%

lhs =1/y(3);
rhs =params(2)/y(3)*(params(1)*exp(y(2))*y(1)^(params(1)-1)+1-params(5));
residual(1)= lhs-rhs;
lhs =y(1);
rhs =(-y(3))+y(1)*(1-params(5))+y(4);
residual(2)= lhs-rhs;
lhs =y(4);
rhs =exp(y(2))*y(1)^params(1);
residual(3)= lhs-rhs;
lhs =y(2);
rhs =y(2)*params(4)+x(1);
residual(4)= lhs-rhs;
if ~isreal(residual)
  residual = real(residual)+imag(residual).^2;
end
if nargout >= 2,
  g1 = zeros(4, 4);

  %
  % Jacobian matrix
  %

  g1(1,1)=(-(params(2)/y(3)*params(1)*exp(y(2))*getPowerDeriv(y(1),params(1)-1,1)));
  g1(1,2)=(-(params(2)/y(3)*params(1)*exp(y(2))*y(1)^(params(1)-1)));
  g1(1,3)=(-1)/(y(3)*y(3))-(params(1)*exp(y(2))*y(1)^(params(1)-1)+1-params(5))*(-params(2))/(y(3)*y(3));
  g1(2,1)=1-(1-params(5));
  g1(2,3)=1;
  g1(2,4)=(-1);
  g1(3,1)=(-(exp(y(2))*getPowerDeriv(y(1),params(1),1)));
  g1(3,2)=(-(exp(y(2))*y(1)^params(1)));
  g1(3,4)=1;
  g1(4,2)=1-params(4);
  if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
  end
end
if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],4,16);
end
end
