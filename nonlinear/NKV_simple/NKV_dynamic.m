function [residual, g1, g2, g3] = NKV_dynamic(y, x, params, steady_state, it_)
%
% Status : Computes dynamic model for Dynare
%
% Inputs :
%   y         [#dynamic variables by 1] double    vector of endogenous variables in the order stored
%                                                 in M_.lead_lag_incidence; see the Manual
%   x         [M_.exo_nbr by nperiods] double     matrix of exogenous variables (in declaration order)
%                                                 for all simulation periods
%   params    [M_.param_nbr by 1] double          vector of parameter values in declaration order
%   it_       scalar double                       time period for exogenous variables for which to evaluate the model
%
% Outputs:
%   residual  [M_.endo_nbr by 1] double    vector of residuals of the dynamic model equations in order of 
%                                          declaration of the equations
%   g1        [M_.endo_nbr by #dynamic variables] double    Jacobian matrix of the dynamic model equations;
%                                                           rows: equations in order of declaration
%                                                           columns: variables in order stored in M_.lead_lag_incidence
%   g2        [M_.endo_nbr by (#dynamic variables)^2] double   Hessian matrix of the dynamic model equations;
%                                                              rows: equations in order of declaration
%                                                              columns: variables in order stored in M_.lead_lag_incidence
%   g3        [M_.endo_nbr by (#dynamic variables)^3] double   Third order derivative matrix of the dynamic model equations;
%                                                              rows: equations in order of declaration
%                                                              columns: variables in order stored in M_.lead_lag_incidence
%
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

%
% Model equations
%

residual = zeros(3, 1);
lhs =y(3);
rhs =y(5)-1/params(13)*(y(1)-y(4))+params(14)*x(it_, 1);
residual(1)= lhs-rhs;
lhs =y(2);
rhs =y(4)*params(2)+y(3)*params(5)+params(18)*x(it_, 2);
residual(2)= lhs-rhs;
lhs =y(1);
rhs =y(2)*params(11)-y(3)*params(12)+params(19)*x(it_, 3);
residual(3)= lhs-rhs;
if nargout >= 2,
  g1 = zeros(3, 8);

  %
  % Jacobian matrix
  %

  g1(1,1)=1/params(13);
  g1(1,4)=(-(1/params(13)));
  g1(1,3)=1;
  g1(1,5)=(-1);
  g1(1,6)=(-params(14));
  g1(2,2)=1;
  g1(2,4)=(-params(2));
  g1(2,3)=(-params(5));
  g1(2,7)=(-params(18));
  g1(3,1)=1;
  g1(3,2)=(-params(11));
  g1(3,3)=params(12);
  g1(3,8)=(-params(19));
end
if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],3,64);
end
if nargout >= 4,
  %
  % Third order derivatives
  %

  g3 = sparse([],[],[],3,512);
end
end
