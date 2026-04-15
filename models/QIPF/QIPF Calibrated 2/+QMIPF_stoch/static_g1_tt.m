function T = static_g1_tt(T, y, x, params)
% function T = static_g1_tt(T, y, x, params)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T         [#temp variables by 1]  double   vector of temporary terms to be filled by function
%   y         [M_.endo_nbr by 1]      double   vector of endogenous variables in declaration order
%   x         [M_.exo_nbr by 1]       double   vector of exogenous variables in declaration order
%   params    [M_.param_nbr by 1]     double   vector of parameter values in declaration order
%
% Output:
%   T         [#temp variables by 1]  double   vector of temporary terms
%

assert(length(T) >= 224);

T = QMIPF_stoch.static_resid_tt(T, y, x, params);

T(159) = getPowerDeriv(y(12)-y(12)*params(87)-params(114)*y(71),T(1),1);
T(160) = getPowerDeriv(y(13)-params(87)*y(13)-params(114)*y(71),T(1),1);
T(161) = getPowerDeriv(y(14)-y(14)*params(88)-params(115)*y(72),T(1),1);
T(162) = getPowerDeriv(y(15)-params(88)*y(15)-params(115)*y(72),T(1),1);
T(163) = y(8)*params(30)*getPowerDeriv(y(29),T(41),1);
T(164) = y(9)*params(30)*getPowerDeriv(y(30),T(41),1);
T(165) = y(10)*params(31)*getPowerDeriv(y(31),T(54),1);
T(166) = y(143)*y(99)*(-y(25))/(y(31)*y(31));
T(167) = y(11)*params(31)*getPowerDeriv(y(32),T(54),1);
T(168) = y(23)*params(33)*getPowerDeriv(y(37),T(43),1);
T(169) = y(23)*params(33)*getPowerDeriv(y(38),T(43),1);
T(170) = y(24)*params(34)*getPowerDeriv(y(39),T(57),1);
T(171) = y(24)*params(34)*getPowerDeriv(y(40),T(57),1);
T(172) = 1/y(87);
T(173) = 1/y(81);
T(174) = getPowerDeriv(y(67),1+params(7),1);
T(175) = getPowerDeriv(y(67)*y(148),1-params(1),1);
T(176) = getPowerDeriv(y(148)*y(68),1-params(1),1);
T(177) = getPowerDeriv(y(69),1+params(7),1);
T(178) = getPowerDeriv(y(69)*y(177),1-params(1),1);
T(179) = getPowerDeriv(y(177)*y(70),1-params(1),1);
T(180) = getPowerDeriv(T(7),T(8),1);
T(181) = getPowerDeriv(T(7),T(11),1);
T(182) = getPowerDeriv(y(126)*T(7),T(11),1);
T(183) = getPowerDeriv(T(16),params(16),1);
T(184) = getPowerDeriv(T(15)*T(17),params(27)*params(20),1);
T(185) = getPowerDeriv(T(77),T(8),1);
T(186) = getPowerDeriv(T(77),T(11),1);
T(187) = getPowerDeriv(y(128)*T(77),T(11),1);
T(188) = (-y(93))/(y(87)*y(87));
T(189) = getPowerDeriv(T(28),(1+params(36))*T(29),1);
T(190) = getPowerDeriv(T(28),T(31),1);
T(191) = getPowerDeriv(T(28)*y(155),T(36),1);
T(192) = getPowerDeriv(T(28)*y(159),T(31),1);
T(193) = (-y(94))/(y(89)*y(89));
T(194) = getPowerDeriv(T(91),T(29)*(1+params(45)),1);
T(195) = getPowerDeriv(T(91),T(93),1);
T(196) = getPowerDeriv(T(91)*y(156),T(98),1);
T(197) = getPowerDeriv(T(91)*y(160),T(93),1);
T(198) = (-y(95))/(y(91)*y(91));
T(199) = getPowerDeriv(y(172)*T(102),T(101),1);
T(200) = getPowerDeriv(T(102),T(29)*(1+params(40)),1);
T(201) = getPowerDeriv(T(102),T(110),1);
T(202) = getPowerDeriv(T(102)*y(176),T(110),1);
T(203) = (-y(96))/(y(92)*y(92));
T(204) = getPowerDeriv(y(171)*T(40),T(39),1);
T(205) = getPowerDeriv(T(40),T(29)*(1+params(39)),1);
T(206) = getPowerDeriv(T(40),T(49),1);
T(207) = getPowerDeriv(T(40)*y(175),T(49),1);
T(208) = 1/y(89);
T(209) = 1/y(91);
T(210) = 1/y(92);
T(211) = (-(1+params(82)))/(y(112)*y(112))/((1+params(82))/y(112));
T(212) = (-(1+params(82)))/(y(113)*y(113))/((1+params(82))/y(113));
T(213) = (-(1+params(82)))/(y(114)*y(114))/((1+params(82))/y(114));
T(214) = (-(1+params(82)))/(y(115)*y(115))/((1+params(82))/y(115));
T(215) = (-(1+params(83)))/(y(116)*y(116))/((1+params(83))/y(116));
T(216) = (-(1+params(83)))/(y(117)*y(117))/((1+params(83))/y(117));
T(217) = getPowerDeriv(y(120),T(23)*(1+params(36)),1);
T(218) = getPowerDeriv(y(121),T(23)*(1+params(39)),1);
T(219) = getPowerDeriv(y(122),T(23)*(1+params(40)),1);
T(220) = getPowerDeriv(y(123),T(23)*(1+params(45)),1);
T(221) = getPowerDeriv(y(132)/y(126),T(8),1);
T(222) = getPowerDeriv(y(133)/y(128),T(8),1);
T(223) = (-(getPowerDeriv(y(148),1-params(1),1)))/(T(21)*T(21));
T(224) = (-(getPowerDeriv(y(177),1-params(1),1)))/(T(85)*T(85));

end
