var 
Pratio Sfunc SfuncD SfuncDflex Sfuncflex a afunc afuncD afuncDflex afuncflex b c cflex dc dinve dp dw dwobs dy epinfma ewma gam1 gam2 gam3 gamw1 gamw2 gamw3 gy inve inveflex k kflex kp kpflex lab labflex labobs mc ms pdot pdotl pinf pinfobs pk pkflex qs qsaux r rk rkflex robs rrflex spinf sw w wdot wdotl wflex wnew xi xiflex y yflex ygap zcap zcapflex ;

varexo 
ea eb eg em epinf eqs ew ;

parameters 
SCALE1_eb SCALE1_epinf SCALE1_eqs SCALE1_ew cZcap calfa cbetabar cfc cg cgamma cgy chabb cindp cindw clandaw cmap cmaw constelab cpie cprobp cprobw crdy crhoa crhob crhog crhoms crhopinf crhoqs crhow crpi crr cry csadjcost csigl csigma ctou ctrend curvP curvW curvp curvw mcflex z_ea z_eb z_eg z_em z_epinf z_eqs z_ew ;

% Parameter definitions:
	ctou	=	0.025;
	clandaw	=	1.5;
	cg	=	0.18;
	curvp	=	10.0;
	curvw	=	10.0;
	calfa	=	0.24;
	csigma	=	1.5;
	cfc	=	1.5;
	cgy	=	0.51;
	csadjcost	=	6.0144;
	chabb	=	0.6361;
	cprobw	=	0.8087;
	csigl	=	1.9423;
	cprobp	=	0.6;
	cindw	=	0.3243;
	cindp	=	0.47;
	czcap	=	0.2696;
	crpi	=	1.488;
	crr	=	0.8762;
	cry	=	0.0593;
	crdy	=	0.2347;
	crhoa	=	0.9977;
	crhob	=	0.5799;
	crhog	=	0.9957;
	crhoqs	=	0.7165;
	crhoms	=	0.0;
	crhopinf	=	0.0;
	crhow	=	0.0;
	cmap	=	0.0;
	cmaw	=	0.0;
	constelab	=	0.0;
	constepinf	=	0.7;
	constebeta	=	0.742;
	ctrend	=	0.3982;
	z_ea	=	0.4618;
	z_eb	=	1.8513;
	z_eg	=	0.609;
	z_em	=	0.2397;
	z_ew	=	0.2089;
	z_eqs	=	0.6017;
	z_epinf	=	0.1455;
	mcflex	=	0.666666666666667;
	cpie	=	1.007;
	cgamma = 1 + ctrend / 100;
	SCALE1_eb = -(((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) ^ -1);
	cbeta = 1 / (1 + constebeta / 100);
	SCALE1_eqs = (cgamma ^ 2 * csadjcost) * (1 + cbeta * cgamma ^ (1 - csigma));
	cbetabar = cbeta * cgamma ^ -csigma;
	SCALE1_epinf = 1 / (((1 / (1 + cbetabar * cgamma * cindp)) * (((1 - cprobp) * (1 - cbetabar * cgamma * cprobp)) / cprobp)) / ((cfc - 1) * curvp + 1));
	SCALE1_ew = 1 / ((((1 - cprobw) * (1 - cbetabar * cgamma * cprobw)) / ((1 + cbetabar * cgamma) * cprobw)) * (1 / ((clandaw - 1) * curvw + 1)));
	cZcap = czcap / (1 - czcap);
	clandap = cfc;
	curvP = (curvp * (1 - clandap)) / clandap;
	curvW = (curvw * (1 - clandaw)) / clandaw;

model;
	y(0) = c(0) + inve(0) + STEADY_STATE(y) * gy(0) + (afunc(0) * kp(-1)) / cgamma;

	(y(0) * (pdot(0) + curvP)) / (1 + curvP) = a(0) * k(0) ^ calfa * lab(0) ^ (1 - calfa) - (cfc - 1) * STEADY_STATE(y);

	k(0) = (kp(-1) * zcap(0)) / cgamma;

	kp(0) = inve(0) * qs(0) * (1 - Sfunc(0)) + (kp(-1) * (1 - ctou)) / cgamma;

	pdot(0) = (1 - cprobp) * (Pratio(0) / dp(0)) ^ ((-cfc * (1 + curvP)) / (cfc - 1)) + cprobp * pdot(-1) * (((dp(-1) / dp(0)) * pinf(-1) ^ cindp * cpie ^ (1 - cindp)) / pinf(0)) ^ ((-cfc * (1 + curvP)) / (cfc - 1));

	wdot(0) = (1 - cprobw) * (wnew(0) / dw(0)) ^ ((-clandaw * (1 + curvW)) / (clandaw - 1)) + cprobw * wdot(-1) * (((dw(-1) / dw(0)) * pinf(-1) ^ cindw * cpie ^ (1 - cindw)) / pinf(0)) ^ ((-clandaw * (1 + curvW)) / (clandaw - 1));

	1 = (1 - cprobp) * (Pratio(0) / dp(0)) ^ (-((1 + curvp * (1 - cfc))) / (cfc - 1)) + cprobp * (((dp(-1) / dp(0)) * pinf(-1) ^ cindp * cpie ^ (1 - cindp)) / pinf(0)) ^ (-((1 + curvp * (1 - cfc))) / (cfc - 1));

	1 = (1 - cprobw) * (wnew(0) / dw(0)) ^ (-((1 + curvw * (1 - clandaw))) / (clandaw - 1)) + cprobw * (((dw(-1) / dw(0)) * pinf(-1) ^ cindw * cpie ^ (1 - cindw)) / pinf(0)) ^ (-((1 + curvw * (1 - clandaw))) / (clandaw - 1));

	1 = (dp(0) * (1 + pdotl(0) * curvP)) / (1 + curvP);

	w(0) = (dw(0) * (1 + curvW * wdotl(0))) / (1 + curvW);

	pdotl(0) = ((1 - cprobp) * Pratio(0)) / dp(0) + ((((cprobp * dp(-1)) / dp(0)) * pinf(-1) ^ cindp * cpie ^ (1 - cindp)) / pinf(0)) * pdotl(-1);

	wdotl(0) = ((1 - cprobw) * wnew(0)) / dw(0) + ((((cprobw * dw(-1)) / dw(0)) * pinf(-1) ^ cindw * cpie ^ (1 - cindw)) / pinf(0)) * wdotl(-1);

	xi(0) = exp(((csigma - 1) / (1 + csigl)) * ((lab(0) * (curvW + wdot(0))) / (1 + curvW)) ^ (1 + csigl)) * (c(0) - (c(-1) * chabb) / cgamma) ^ -csigma;

	1 = qs(0) * pk(0) * ((1 - Sfunc(0)) - (cgamma * inve(0) * SfuncD(0)) / inve(-1)) + ((SfuncD(1) * xi(1)) / xi(0)) * qsaux(0) * pk(1) * ((cgamma * inve(1)) / inve(0)) ^ 2 * cbetabar;

	xi(0) = (xi(1) * b(0) * r(0) * cbetabar) / pinf(1);

	rk(0) = afuncD(0);

	pk(0) = (((rk(1) * zcap(1) - afunc(1)) + (1 - ctou) * pk(1)) * xi(1) * cbetabar) / xi(0);

	k(0) = ((lab(0) * w(0) * calfa) / (1 - calfa)) / rk(0);

	mc(0) = (w(0) ^ (1 - calfa) * rk(0) ^ calfa) / (a(0) * calfa ^ calfa * (1 - calfa) ^ (1 - calfa));

	(wnew(0) * gamw1(0) * (1 + curvw * (1 - clandaw))) / (1 + curvW) = clandaw * gamw2(0) + ((gamw3(0) * curvW * (clandaw - 1)) / (1 + curvW)) * wnew(0) ^ (1 + (clandaw * (1 + curvW)) / (clandaw - 1));

	gamw1(0) = lab(0) * dw(0) ^ ((clandaw * (1 + curvW)) / (clandaw - 1)) + ((gamw1(1) * ((cpie ^ (1 - cindw) * pinf(0) ^ cindw) / pinf(1)) ^ (-((1 + curvw * (1 - clandaw))) / (clandaw - 1)) * xi(1)) / xi(0)) * cgamma * cprobw * cbetabar;

	gamw2(0) = (c(0) - (c(-1) * chabb) / cgamma) * lab(0) * sw(0) * dw(0) ^ ((clandaw * (1 + curvW)) / (clandaw - 1)) * ((lab(0) * (curvW + wdot(0))) / (1 + curvW)) ^ csigl + ((gamw2(1) * ((cpie ^ (1 - cindw) * pinf(0) ^ cindw) / pinf(1)) ^ ((-clandaw * (1 + curvW)) / (clandaw - 1)) * xi(1)) / xi(0)) * cgamma * cprobw * cbetabar;

	gamw3(0) = lab(0) + ((((gamw3(1) * cpie ^ (1 - cindw) * pinf(0) ^ cindw) / pinf(1)) * xi(1)) / xi(0)) * cgamma * cprobw * cbetabar;

	(Pratio(0) * gam1(0) * (1 + curvp * (1 - cfc))) / (1 + curvP) = cfc * gam2(0) + ((gam3(0) * (cfc - 1) * curvP) / (1 + curvP)) * Pratio(0) ^ (1 + (cfc * (1 + curvP)) / (cfc - 1));

	gam1(0) = y(0) * dp(0) ^ ((cfc * (1 + curvP)) / (cfc - 1)) + ((gam1(1) * xi(1)) / xi(0)) * cgamma * cprobp * cbetabar * ((cpie ^ (1 - cindp) * pinf(0) ^ cindp) / pinf(1)) ^ (-((1 + curvp * (1 - cfc))) / (cfc - 1));

	gam2(0) = y(0) * mc(0) * spinf(0) * dp(0) ^ ((cfc * (1 + curvP)) / (cfc - 1)) + ((gam2(1) * xi(1)) / xi(0)) * cgamma * cprobp * cbetabar * ((cpie ^ (1 - cindp) * pinf(0) ^ cindp) / pinf(1)) ^ ((-cfc * (1 + curvP)) / (cfc - 1));

	gam3(0) = y(0) + ((((gam3(1) * cpie ^ (1 - cindp) * pinf(0) ^ cindp) / pinf(1)) * xi(1)) / xi(0)) * cgamma * cprobp * cbetabar;

	qsaux(0) = qs(1);

	r(0) = STEADY_STATE(r) ^ (1 - crr) * r(-1) ^ crr * (pinf(0) / cpie) ^ ((1 - crr) * crpi) * (y(0) / yflex(0)) ^ ((1 - crr) * cry) * ((y(0) / yflex(0)) / (y(-1) / yflex(-1))) ^ crdy * ms(0);

	afunc(0) = ((STEADY_STATE(rk) * 1) / cZcap) * (exp(cZcap * (zcap(0) - 1)) - 1);

	afuncD(0) = STEADY_STATE(rk) * exp(cZcap * (zcap(0) - 1));

	Sfunc(0) = (csadjcost / 2) * ((cgamma * inve(0)) / inve(-1) - cgamma) ^ 2;

	SfuncD(0) = csadjcost * ((cgamma * inve(0)) / inve(-1) - cgamma);

	a(0) = (1 - crhoa) + crhoa * a(-1) + (z_ea / 100) * ea;

	b(0) = (1 - crhob) + crhob * b(-1) + (z_eb / 100) * SCALE1_eb * eb;

	gy(0) - cg = crhog * (gy(-1) - cg) + (z_eg / 100) * eg + (z_ea / 100) * ea * cgy;

	qs(0) = (1 - crhoqs) + crhoqs * qs(-1) + (z_eqs / 100) * SCALE1_eqs * eqs;

	ms(0) = (1 - crhoms) + crhoms * ms(-1) + (z_em / 100) * em;

	spinf(0) = ((1 - crhopinf) + crhopinf * spinf(-1) + epinfma(0)) - cmap * epinfma(-1);

	epinfma(0) = (z_epinf / 100) * SCALE1_epinf * epinf;

	sw(0) = ((1 - crhow) + crhow * sw(-1) + ewma(0)) - cmaw * ewma(-1);

	ewma(0) = (z_ew / 100) * SCALE1_ew * ew;

	yflex(0) = cflex(0) + inveflex(0) + gy(0) * STEADY_STATE(yflex) + (afuncflex(0) * kpflex(-1)) / cgamma;

	yflex(0) = a(0) * kflex(0) ^ calfa * labflex(0) ^ (1 - calfa) - (cfc - 1) * STEADY_STATE(yflex);

	kflex(0) = (kpflex(-1) * zcapflex(0)) / cgamma;

	kpflex(0) = inveflex(0) * qs(0) * (1 - Sfuncflex(0)) + (kpflex(-1) * (1 - ctou)) / cgamma;

	xiflex(0) = exp(((csigma - 1) / (1 + csigl)) * labflex(0) ^ (1 + csigl)) * (cflex(0) - (cflex(-1) * chabb) / cgamma) ^ -csigma;

	1 = qs(0) * pkflex(0) * ((1 - Sfuncflex(0)) - (cgamma * inveflex(0) * SfuncDflex(0)) / inveflex(-1)) + ((SfuncDflex(1) * qsaux(0) * xiflex(1)) / xiflex(0)) * pkflex(1) * ((cgamma * inveflex(1)) / inveflex(0)) ^ 2 * cbetabar;

	xiflex(0) = xiflex(1) * b(0) * rrflex(0) * cbetabar;

	rkflex(0) = afuncDflex(0);

	pkflex(0) = (((rkflex(1) * zcapflex(1) - afuncflex(1)) + (1 - ctou) * pkflex(1)) * xiflex(1) * cbetabar) / xiflex(0);

	kflex(0) = (((labflex(0) * calfa) / (1 - calfa)) * wflex(0)) / rkflex(0);

	mcflex = (wflex(0) ^ (1 - calfa) * rkflex(0) ^ calfa) / (a(0) * calfa ^ calfa * (1 - calfa) ^ (1 - calfa));

	(wflex(0) * (1 + curvw * (1 - clandaw))) / (1 + curvW) = STEADY_STATE(sw) * (labflex(0) ^ csigl * clandaw * (cflex(0) - (cflex(-1) * chabb) / cgamma) + (wflex(0) * curvW * (clandaw - 1)) / (1 + curvW));

	afuncflex(0) = ((STEADY_STATE(rkflex) * 1) / cZcap) * (exp(cZcap * (zcapflex(0) - 1)) - 1);

	afuncDflex(0) = STEADY_STATE(rkflex) * exp(cZcap * (zcapflex(0) - 1));

	Sfuncflex(0) = (csadjcost / 2) * ((cgamma * inveflex(0)) / inveflex(-1) - cgamma) ^ 2;

	SfuncDflex(0) = csadjcost * ((cgamma * inveflex(0)) / inveflex(-1) - cgamma);

	ygap(0) = 100 * log(y(0) / yflex(0));

	dy(0) = ctrend + 100 * (y(0) / y(-1) - 1);

	dc(0) = ctrend + 100 * (c(0) / c(-1) - 1);

	dinve(0) = ctrend + 100 * (inve(0) / inve(-1) - 1);

	pinfobs(0) = 100 * (pinf(0) - 1);

	robs(0) = 100 * (r(0) - 1);

	dwobs(0) = ctrend + 100 * (w(0) / w(-1) - 1);

	labobs(0) = constelab + 100 * (lab(0) / STEADY_STATE(lab) - 1);

end;

shocks;
var	ea	=	1;
var	eb	=	1;
var	eg	=	1;
var	em	=	1;
var	epinf	=	1;
var	eqs	=	1;
var	ew	=	1;
end;

initval;
	Pratio	=	1.0000000000000002;
	Sfunc	=	0.0;
	SfuncD	=	0.0;
	SfuncDflex	=	0.0;
	Sfuncflex	=	0.0;
	a	=	1.0;
	afunc	=	-3.246291276182176e-18;
	afuncD	=	0.038443305932122175;
	afuncDflex	=	0.0384433059321223;
	afuncflex	=	-3.2462912974120005e-18;
	b	=	1.0;
	c	=	0.8688510496768551;
	cflex	=	0.868851049676854;
	dc	=	0.3982;
	dinve	=	0.3982;
	dp	=	1.0000000000000004;
	dw	=	0.794859774135622;
	dwobs	=	0.3982;
	dy	=	0.3982;
	epinfma	=	0.0;
	ewma	=	0.0;
	gam1	=	3.3519668279713013;
	gam2	=	2.234644551980868;
	gam3	=	3.351966827971306;
	gamw1	=	32.610945351068715;
	gamw2	=	17.280752437399723;
	gamw3	=	6.53727901827707;
	gy	=	0.18;
	inve	=	0.2459903726375893;
	inveflex	=	0.24599037263759022;
	k	=	8.48769486707575;
	kflex	=	8.487694867075723;
	kp	=	8.521492868036443;
	kpflex	=	8.521492868036416;
	lab	=	1.2999370371078451;
	labflex	=	1.2999370371078465;
	labobs	=	0.0;
	mc	=	0.666666666666667;
	ms	=	1.0;
	pdot	=	0.9999999999999993;
	pdotl	=	0.9999999999999999;
	pinf	=	1.007;
	pinfobs	=	0.6999999999999886;
	pk	=	1.0;
	pkflex	=	1.0;
	qs	=	1.0;
	qsaux	=	1.0;
	r	=	1.020537409073647;
	rk	=	0.038443305932122175;
	rkflex	=	0.0384433059321223;
	robs	=	2.0537409073646984;
	rrflex	=	1.0134433059321224;
	spinf	=	1.0;
	sw	=	1.0;
	w	=	0.7948597741356216;
	wdot	=	1.0000000000000002;
	wdotl	=	0.9999999999999996;
	wflex	=	0.7948597741356209;
	wnew	=	0.7948597741356223;
	xi	=	8.040664434502393;
	xiflex	=	8.04066443450242;
	y	=	1.3595627101395662;
	yflex	=	1.3595627101395662;
	ygap	=	0.0;
	zcap	=	1.0;
	zcapflex	=	1.0;
end;

stoch_simul(order = 1, irf = 40);

% Extended Path (SEP) tests similar to Dynare's rs.mod
% Test 1: SEP with order=1 (periods=10, branching length = 1)
disp('================================================================');
disp('DYNARE EXTENDED PATH TEST 1: periods=10, order=1');
disp('================================================================');
extended_path(periods=10, order=1);

% Report key steady state values for comparison
disp(' ');
disp('Key steady state values (SEP order=1):');
disp(['  y    = ' num2str(oo_.endo_simul(strmatch('y', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);
disp(['  c    = ' num2str(oo_.endo_simul(strmatch('c', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);
disp(['  inve = ' num2str(oo_.endo_simul(strmatch('inve', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);
disp(['  pinf = ' num2str(oo_.endo_simul(strmatch('pinf', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);
disp(['  lab  = ' num2str(oo_.endo_simul(strmatch('lab', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);

% Test 2: SEP with order=2 (periods=10, branching length = 2)
disp(' ');
disp('================================================================');
disp('DYNARE EXTENDED PATH TEST 2: periods=10, order=2');
disp('================================================================');
extended_path(periods=10, order=2);

% Report key steady state values for comparison  
disp(' ');
disp('Key steady state values (SEP order=2):');
disp(['  y    = ' num2str(oo_.endo_simul(strmatch('y', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);
disp(['  c    = ' num2str(oo_.endo_simul(strmatch('c', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);
disp(['  inve = ' num2str(oo_.endo_simul(strmatch('inve', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);
disp(['  pinf = ' num2str(oo_.endo_simul(strmatch('pinf', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);
disp(['  lab  = ' num2str(oo_.endo_simul(strmatch('lab', M_.endo_names, 'exact'), M_.maximum_lag+1), '%12.8f')]);

disp(' ');
disp('================================================================');
disp('EXTENDED PATH TESTS COMPLETE');
disp('================================================================');
