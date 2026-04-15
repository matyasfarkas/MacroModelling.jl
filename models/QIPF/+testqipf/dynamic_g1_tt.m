function [T_order, T] = dynamic_g1_tt(y, x, params, steady_state, T_order, T)
if T_order >= 1
    return
end
[T_order, T] = testqipf.dynamic_resid_tt(y, x, params, steady_state, T_order, T);
T_order = 1;
if size(T, 1) < 171
    T = [T; NaN(171 - size(T, 1), 1)];
end
T(131) = getPowerDeriv(y(129)-params(87)*y(8)-params(112)*y(159),T(1),1);
T(132) = getPowerDeriv(y(130)-params(88)*y(9)-params(113)*y(160),T(1),1);
T(133) = y(127)*params(30)*getPowerDeriv(y(137),T(43),1);
T(134) = y(128)*params(31)*getPowerDeriv(y(138),(1+params(50))/params(50),1);
T(135) = y(133)*params(33)*getPowerDeriv(y(141),T(45),1);
T(136) = y(134)*params(34)*getPowerDeriv(y(142),(1+params(54))/params(54),1);
T(137) = 1/y(171);
T(138) = getPowerDeriv(y(157)*y(212),1-params(1),1);
T(139) = getPowerDeriv(y(158)*y(241),1-params(1),1);
T(140) = getPowerDeriv(T(128),params(16),1);
T(141) = getPowerDeriv(T(17)*T(129),params(27)*params(20),1);
T(142) = getPowerDeriv(T(14),T(12),1);
T(143) = getPowerDeriv(T(18),T(9),1);
T(144) = getPowerDeriv(T(8),T(9),1);
T(145) = getPowerDeriv(T(8),T(12),1);
T(146) = getPowerDeriv(T(78),T(12),1);
T(147) = getPowerDeriv(T(82),T(9),1);
T(148) = getPowerDeriv(T(74),T(9),1);
T(149) = getPowerDeriv(T(74),T(12),1);
T(150) = getPowerDeriv(T(38)*y(98),T(37),1);
T(151) = getPowerDeriv(T(38)*y(102),T(32),1);
T(152) = getPowerDeriv(T(29),(1+params(36))*T(30),1);
T(153) = getPowerDeriv(T(29),T(32),1);
T(154) = getPowerDeriv(T(99)*y(99),T(98),1);
T(155) = getPowerDeriv(T(99)*y(103),T(93),1);
T(156) = getPowerDeriv(T(91),T(30)*(1+params(45)),1);
T(157) = getPowerDeriv(T(91),T(93),1);
T(158) = getPowerDeriv(T(103)*y(115),T(102),1);
T(159) = getPowerDeriv(T(103)*y(119),T(112),1);
T(160) = getPowerDeriv(T(109),T(30)*(1+params(40)),1);
T(161) = getPowerDeriv(T(109),T(112),1);
T(162) = getPowerDeriv(T(42)*y(114),T(41),1);
T(163) = getPowerDeriv(T(42)*y(118),T(52),1);
T(164) = getPowerDeriv(T(50),T(30)*(1+params(39)),1);
T(165) = getPowerDeriv(T(50),T(52),1);
T(166) = getPowerDeriv(y(196),T(24)*(1+params(36)),1);
T(167) = getPowerDeriv(y(197),T(24)*(1+params(39)),1);
T(168) = getPowerDeriv(y(198),T(24)*(1+params(40)),1);
T(169) = getPowerDeriv(y(199),T(24)*(1+params(45)),1);
T(170) = getPowerDeriv(y(204)/y(202),T(9),1);
T(171) = getPowerDeriv(y(205)/y(203),T(9),1);
end
