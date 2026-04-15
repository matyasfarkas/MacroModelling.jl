function T = dynamic_g1_tt(T, y, x, params, steady_state, it_)
% function T = dynamic_g1_tt(T, y, x, params, steady_state, it_)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T             [#temp variables by 1]     double  vector of temporary terms to be filled by function
%   y             [#dynamic variables by 1]  double  vector of endogenous variables in the order stored
%                                                    in M_.lead_lag_incidence; see the Manual
%   x             [nperiods by M_.exo_nbr]   double  matrix of exogenous variables (in declaration order)
%                                                    for all simulation periods
%   steady_state  [M_.endo_nbr by 1]         double  vector of steady state values
%   params        [M_.param_nbr by 1]        double  vector of parameter values in declaration order
%   it_           scalar                     double  time period for exogenous variables for which
%                                                    to evaluate the model
%
% Output:
%   T           [#temp variables by 1]       double  vector of temporary terms
%

assert(length(T) >= 234);

T = QMIPF_stoch.dynamic_resid_tt(T, y, x, params, steady_state, it_);

T(181) = getPowerDeriv(y(97)-params(87)*y(7)-params(114)*y(156),T(1),1);
T(182) = getPowerDeriv(y(98)-params(87)*y(8)-params(114)*y(156),T(1),1);
T(183) = getPowerDeriv(y(99)-params(88)*y(9)-params(115)*y(157),T(1),1);
T(184) = getPowerDeriv(y(100)-params(88)*y(10)-params(115)*y(157),T(1),1);
T(185) = y(93)*params(30)*getPowerDeriv(y(114),T(43),1);
T(186) = y(94)*params(30)*getPowerDeriv(y(115),T(43),1);
T(187) = y(95)*params(31)*getPowerDeriv(y(116),T(57),1);
T(188) = y(228)*y(184)*(-y(110))/(y(116)*y(116));
T(189) = y(96)*params(31)*getPowerDeriv(y(117),T(57),1);
T(190) = y(108)*params(33)*getPowerDeriv(y(122),T(45),1);
T(191) = y(108)*params(33)*getPowerDeriv(y(123),T(45),1);
T(192) = y(109)*params(34)*getPowerDeriv(y(124),T(60),1);
T(193) = y(109)*params(34)*getPowerDeriv(y(125),T(60),1);
T(194) = 1/y(172);
T(195) = getPowerDeriv(y(152),1+params(7),1);
T(196) = getPowerDeriv(y(152)*y(233),1-params(1),1);
T(197) = getPowerDeriv(y(233)*y(153),1-params(1),1);
T(198) = getPowerDeriv(y(154),1+params(7),1);
T(199) = getPowerDeriv(y(154)*y(262),1-params(1),1);
T(200) = getPowerDeriv(y(262)*y(155),1-params(1),1);
T(201) = getPowerDeriv(T(178),params(16),1);
T(202) = getPowerDeriv(T(17)*T(179),params(27)*params(20),1);
T(203) = getPowerDeriv(T(14),T(12),1);
T(204) = getPowerDeriv(T(18),T(9),1);
T(205) = getPowerDeriv(T(8),T(9),1);
T(206) = getPowerDeriv(T(8),T(12),1);
T(207) = getPowerDeriv(T(85),T(12),1);
T(208) = getPowerDeriv(T(89),T(9),1);
T(209) = getPowerDeriv(T(81),T(9),1);
T(210) = getPowerDeriv(T(81),T(12),1);
T(211) = getPowerDeriv(T(38)*y(58),T(37),1);
T(212) = getPowerDeriv(T(38)*y(62),T(32),1);
T(213) = getPowerDeriv(T(29),(1+params(36))*T(30),1);
T(214) = getPowerDeriv(T(29),T(32),1);
T(215) = getPowerDeriv(T(106)*y(59),T(105),1);
T(216) = getPowerDeriv(T(106)*y(63),T(100),1);
T(217) = getPowerDeriv(T(98),T(30)*(1+params(45)),1);
T(218) = getPowerDeriv(T(98),T(100),1);
T(219) = getPowerDeriv(T(110)*y(65),T(109),1);
T(220) = getPowerDeriv(T(110)*y(69),T(119),1);
T(221) = getPowerDeriv(T(116),T(30)*(1+params(40)),1);
T(222) = getPowerDeriv(T(116),T(119),1);
T(223) = getPowerDeriv(T(42)*y(64),T(41),1);
T(224) = getPowerDeriv(T(42)*y(68),T(52),1);
T(225) = getPowerDeriv(T(50),T(30)*(1+params(39)),1);
T(226) = getPowerDeriv(T(50),T(52),1);
T(227) = getPowerDeriv(y(205),T(24)*(1+params(36)),1);
T(228) = getPowerDeriv(y(206),T(24)*(1+params(39)),1);
T(229) = getPowerDeriv(y(207),T(24)*(1+params(40)),1);
T(230) = getPowerDeriv(y(208),T(24)*(1+params(45)),1);
T(231) = getPowerDeriv(y(217)/y(211),T(9),1);
T(232) = getPowerDeriv(y(218)/y(213),T(9),1);
T(233) = (-(getPowerDeriv(y(233),1-params(1),1)))/(T(22)*T(22));
T(234) = (-(getPowerDeriv(y(262),1-params(1),1)))/(T(92)*T(92));

end
