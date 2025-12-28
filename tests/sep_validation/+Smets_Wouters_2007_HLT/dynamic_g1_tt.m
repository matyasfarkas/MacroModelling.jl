function [T_order, T] = dynamic_g1_tt(y, x, params, steady_state, T_order, T)
if T_order >= 1
    return
end
[T_order, T] = Smets_Wouters_2007_HLT.dynamic_resid_tt(y, x, params, steady_state, T_order, T);
T_order = 1;
if size(T, 1) < 106
    T = [T; NaN(106 - size(T, 1), 1)];
end
T(72) = getPowerDeriv(T(3),T(4),1);
T(73) = getPowerDeriv(T(3),T(15),1);
T(74) = (-(params(12)/params(10)));
T(75) = getPowerDeriv(T(22),(-params(35)),1);
T(76) = getPowerDeriv(T(63),(-params(35)),1);
T(77) = T(6)*T(5)*1/y(82)/y(108);
T(78) = getPowerDeriv(T(7),T(4),1);
T(79) = getPowerDeriv(T(7),T(15),1);
T(80) = T(6)*T(5)*(-y(16))/(y(82)*y(82))/y(108);
T(81) = getPowerDeriv(y(82),T(41),1);
T(82) = T(12)*T(11)*1/y(83)/y(108);
T(83) = getPowerDeriv(T(13),T(10),1);
T(84) = getPowerDeriv(T(13),T(16),1);
T(85) = getPowerDeriv(T(9),T(10),1);
T(86) = T(12)*T(11)*(-y(17))/(y(83)*y(83))/y(108);
T(87) = getPowerDeriv(T(9),T(16),1);
T(88) = getPowerDeriv(y(83),T(33),1);
T(89) = getPowerDeriv(T(20),1+params(34),1);
T(90) = getPowerDeriv(T(20),params(34),1);
T(91) = getPowerDeriv(y(42),params(13),1);
T(92) = T(6)*y(16)/y(82)*T(91)/y(108);
T(93) = getPowerDeriv(y(42),params(14),1);
T(94) = T(12)*y(17)/y(83)*T(93)/y(108);
T(95) = (-(y(16)/y(82)*T(5)*T(6)))/(y(108)*y(108));
T(96) = (-(y(17)/y(83)*T(11)*T(12)))/(y(108)*y(108));
T(97) = getPowerDeriv(y(108),params(14),1);
T(98) = T(12)*T(97)/y(174);
T(99) = getPowerDeriv(T(37),T(16),1);
T(100) = getPowerDeriv(T(37),T(10),1);
T(101) = getPowerDeriv(y(108),params(13),1);
T(102) = T(6)*T(101)/y(174);
T(103) = getPowerDeriv(T(46),T(15),1);
T(104) = getPowerDeriv(T(46),T(4),1);
T(105) = getPowerDeriv(T(56),params(22),1);
T(106) = getPowerDeriv(T(53),(1-params(31))*params(32),1);
end
