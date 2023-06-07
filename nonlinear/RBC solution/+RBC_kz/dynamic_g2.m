function g2 = dynamic_g2(T, y, x, params, steady_state, it_, T_flag)
% function g2 = dynamic_g2(T, y, x, params, steady_state, it_, T_flag)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T             [#temp variables by 1]     double   vector of temporary terms to be filled by function
%   y             [#dynamic variables by 1]  double   vector of endogenous variables in the order stored
%                                                     in M_.lead_lag_incidence; see the Manual
%   x             [nperiods by M_.exo_nbr]   double   matrix of exogenous variables (in declaration order)
%                                                     for all simulation periods
%   steady_state  [M_.endo_nbr by 1]         double   vector of steady state values
%   params        [M_.param_nbr by 1]        double   vector of parameter values in declaration order
%   it_           scalar                     double   time period for exogenous variables for which
%                                                     to evaluate the model
%   T_flag        boolean                    boolean  flag saying whether or not to calculate temporary terms
%
% Output:
%   g2
%

if T_flag
    T = RBC_kz.dynamic_g2_tt(T, y, x, params, steady_state, it_);
end
g2_i = zeros(12,1);
g2_j = zeros(12,1);
g2_v = zeros(12,1);

g2_i(1)=1;
g2_i(2)=1;
g2_i(3)=1;
g2_i(4)=1;
g2_i(5)=1;
g2_i(6)=1;
g2_i(7)=1;
g2_i(8)=1;
g2_i(9)=1;
g2_i(10)=1;
g2_i(11)=1;
g2_i(12)=1;
g2_j(1)=1;
g2_j(2)=2;
g2_j(3)=7;
g2_j(4)=15;
g2_j(5)=17;
g2_j(6)=27;
g2_j(7)=16;
g2_j(8)=21;
g2_j(9)=28;
g2_j(10)=23;
g2_j(11)=8;
g2_j(12)=22;
g2_v(1)=exp(y(2))*getPowerDeriv(y(1),params(1),2);
g2_v(2)=T(6);
g2_v(3)=g2_v(2);
g2_v(4)=(-((T(5)*T(5)*(T(8)*T(9)+T(5)*exp(y(4))*getPowerDeriv(y(3),params(1),2)-(T(8)*T(9)+T(3)*params(2)*params(1)*exp(y(4))*getPowerDeriv(y(3),params(1)-1,2)))-(T(5)*T(8)-T(3)*T(9))*(T(5)*T(9)+T(5)*T(9)))/(T(5)*T(5)*T(5)*T(5))));
g2_v(5)=(-(T(9)/(T(5)*T(5))));
g2_v(6)=g2_v(5);
g2_v(7)=(-((T(5)*T(5)*(T(8)*T(4)*params(2)+T(5)*T(7)-(T(3)*T(9)+T(2)*T(9)))-(T(5)*T(8)-T(3)*T(9))*(T(5)*T(4)*params(2)+T(5)*T(4)*params(2)))/(T(5)*T(5)*T(5)*T(5))));
g2_v(8)=g2_v(7);
g2_v(9)=(-(T(4)*params(2)/(T(5)*T(5))));
g2_v(10)=g2_v(9);
g2_v(11)=T(1);
g2_v(12)=(-((T(5)*T(5)*(T(2)*T(5)+T(2)*T(4)*params(2)-(T(3)*T(4)*params(2)+T(2)*T(4)*params(2)))-(T(2)*T(5)-T(3)*T(4)*params(2))*(T(5)*T(4)*params(2)+T(5)*T(4)*params(2)))/(T(5)*T(5)*T(5)*T(5))));
g2 = sparse(g2_i,g2_j,g2_v,2,36);
end
