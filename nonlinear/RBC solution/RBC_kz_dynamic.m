function [residual, g1, g2, g3] = RBC_kz_dynamic(y, x, params, steady_state, it_)
%
% Status : Computes dynamic model for Dynare
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

%
% Model equations
%

residual = zeros(2, 1);
T29 = (-y(5))+y(3)*(1-params(5))+exp(y(4))*y(3)^params(1);
T36 = (1-params(5)+params(1)*exp(y(4))*y(3)^(params(1)-1))*params(2);
lhs =(-y(3))+(1-params(5))*y(1)+exp(y(2))*y(1)^params(1);
rhs =T29/T36;
residual(1)= lhs-rhs;
lhs =y(4);
rhs =y(2)*params(4)+params(3)*x(it_, 1);
residual(2)= lhs-rhs;
if nargout >= 2,
  g1 = zeros(2, 6);

  %
  % Jacobian matrix
  %

  g1(1,1)=1-params(5)+exp(y(2))*getPowerDeriv(y(1),params(1),1);
  g1(1,3)=(-1)-(T36*(1-params(5)+exp(y(4))*getPowerDeriv(y(3),params(1),1))-T29*params(2)*params(1)*exp(y(4))*getPowerDeriv(y(3),params(1)-1,1))/(T36*T36);
  g1(1,5)=(-((-1)/T36));
  g1(1,2)=exp(y(2))*y(1)^params(1);
  g1(1,4)=(-((exp(y(4))*y(3)^params(1)*T36-T29*params(1)*exp(y(4))*y(3)^(params(1)-1)*params(2))/(T36*T36)));
  g1(2,2)=(-params(4));
  g1(2,4)=1;
  g1(2,6)=(-params(3));
end
if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],2,36);
end
if nargout >= 4,
  %
  % Third order derivatives
  %

  g3 = sparse([],[],[],2,216);
end
end
