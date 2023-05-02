function [residual, g1, g2] = RBC_kz_static(y, x, params)
%
% Status : Computes static model for Dynare
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

residual = zeros( 2, 1);

%
% Model equations
%

T18 = exp(y(2))*y(1)^params(1);
T19 = (-y(1))+y(1)*(1-params(5))+T18;
T26 = (1-params(5)+exp(y(2))*params(1)*y(1)^(params(1)-1))*params(2);
lhs =T19;
rhs =T19/T26-0.00265085515766751*x(1);
residual(1)= lhs-rhs;
lhs =y(2);
rhs =y(2)*params(4)+x(1)*params(3);
residual(2)= lhs-rhs;
if ~isreal(residual)
  residual = real(residual)+imag(residual).^2;
end
if nargout >= 2,
  g1 = zeros(2, 2);

  %
  % Jacobian matrix
  %

  g1(1,1)=(-1)+1-params(5)+exp(y(2))*getPowerDeriv(y(1),params(1),1)-(T26*((-1)+1-params(5)+exp(y(2))*getPowerDeriv(y(1),params(1),1))-T19*params(2)*exp(y(2))*params(1)*getPowerDeriv(y(1),params(1)-1,1))/(T26*T26);
  g1(1,2)=T18-(T18*T26-T19*exp(y(2))*params(1)*y(1)^(params(1)-1)*params(2))/(T26*T26);
  g1(2,2)=1-params(4);
  if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
  end
end
if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],2,4);
end
end
