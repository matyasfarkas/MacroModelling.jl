# HLT model variant: Rotemberg (1982) quadratic adjustment-cost pricing
# Replaces Calvo staggered pricing with quadratic inflation cost.
# Eliminates 7 pricing variables (pdot, pdotl, dp, Pratio, gam1, gam2, gam3).
# Adds explicit pi^2 nonlinearity in the resource constraint.
# phi_R calibrated to match baseline Calvo slope at cprobp=0.6.
# Wage block remains Calvo+Kimball (unchanged).

@model Smets_Wouters_2007_HLT_obc_Rot begin
	# Resource constraint with Rotemberg adjustment cost
	y[0] = c[0] + inve[0] + y[ss] * gy[0] + afunc[0] * kp[-1] / cgamma + phi_R / 2 * (pinf[0] - cpie) ^ 2 * y[0]

	# Production function (no price dispersion under Rotemberg)
	y[0] = a[0] * k[0] ^ calfa * lab[0] ^ (1 - calfa) - (cfc - 1) * y[ss]

	k[0] = kp[-1] * zcap[0] / cgamma

	kp[0] = inve[0] * qs[0] * (1 - Sfunc[0]) + kp[-1] * (1 - ctou) / cgamma

	# Rotemberg NKPC (replaces entire Calvo pricing block)
	# ceps_p = cfc/(cfc-1) is the demand elasticity; at SS pinf=cpie → mc_ss = (ceps_p-1)/ceps_p = 1/cfc
	phi_R * pinf[0] * (pinf[0] - cpie) = (1 - ceps_p) + ceps_p * mc[0] * spinf[0] + cbetabar * xi[1] / xi[0] * phi_R * pinf[1] * (pinf[1] - cpie) * y[1] / y[0]

	# HLT Kimball wage block (unchanged from baseline)
	wdot[0] = (1 - cprobw) * (wnew[0] / dw[0]) ^ ((-clandaw) * (1 + curvW) / (clandaw - 1)) +
	          cprobw * wdot[-1] * (dw[-1] / dw[0] * pinf[-1] ^ cindw * cpie ^ (1 - cindw) / pinf[0]) ^ ((-clandaw) * (1 + curvW) / (clandaw - 1))

	# Kimball wage reset condition
	1 = (1 - cprobw) * (wnew[0] / dw[0]) ^ ((-(1 + curvw * (1 - clandaw))) / (clandaw - 1)) +
	    cprobw * (dw[-1] / dw[0] * pinf[-1] ^ cindw * cpie ^ (1 - cindw) / pinf[0]) ^ ((-(1 + curvw * (1 - clandaw))) / (clandaw - 1))

	w[0] = dw[0] * (1 + curvW * wdotl[0]) / (1 + curvW)

	wdotl[0] = (1 - cprobw) * wnew[0] / dw[0] + cprobw * dw[-1] / dw[0] * pinf[-1] ^ cindw * cpie ^ (1 - cindw) / pinf[0] * wdotl[-1]

	xi[0] = exp((csigma - 1) / (1 + csigl) * (lab[0] * (curvW + wdot[0]) / (1 + curvW)) ^ (1 + csigl)) * (c[0] - c[-1] * chabb / cgamma) ^ (-csigma)

	1 = qs[0] * pk[0] * (1 - Sfunc[0] - cgamma * inve[0] * SfuncD[0] / inve[-1]) + SfuncD[1] * xi[1] / xi[0] * qsaux[0] * pk[1] * (cgamma * inve[1] / inve[0]) ^ 2 * cbetabar

	xi[0] = xi[1] * b[0] * r[0] * cbetabar / pinf[1]

	rk[0] = afuncD[0]

	pk[0] = (rk[1] * zcap[1] - afunc[1] + (1 - ctou) * pk[1]) * xi[1] * cbetabar / xi[0]

	k[0] = lab[0] * w[0] * calfa / (1 - calfa) / rk[0]

	mc[0] = w[0] ^ (1 - calfa) * rk[0] ^ calfa / (a[0] * calfa ^ calfa * (1 - calfa) ^ (1 - calfa))

	wnew[0] * gamw1[0] * (1 + curvw * (1 - clandaw)) / (1 + curvW) = clandaw * gamw2[0] + gamw3[0] * curvW * (clandaw - 1) / (1 + curvW) * wnew[0] ^ (1 + clandaw * (1 + curvW) / (clandaw - 1))

	gamw1[0] = lab[0] * dw[0] ^ (clandaw * (1 + curvW) / (clandaw - 1)) + gamw1[1] * (cpie ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - (1 + curvw * (1 - clandaw))) / (clandaw - 1)) * xi[1] / xi[0] * cgamma * cprobw * cbetabar

	gamw2[0] = (c[0] - c[-1] * chabb / cgamma) * lab[0] * sw[0] * dw[0] ^ (clandaw * (1 + curvW) / (clandaw - 1)) * (lab[0] * (curvW + wdot[0]) / (1 + curvW)) ^ csigl + gamw2[1] * (cpie ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1]) ^ (( - clandaw) * (1 + curvW) / (clandaw - 1)) * xi[1] / xi[0] * cgamma * cprobw * cbetabar

	gamw3[0] = lab[0] + gamw3[1] * cpie ^ (1 - cindw) * pinf[0] ^ cindw / pinf[1] * xi[1] / xi[0] * cgamma * cprobw * cbetabar

	qsaux[0] = qs[1]

	# Taylor rule with ZLB via OBC (log form)
	log(r[0]) = max(R_bar,
	                (1 - crr) * log(r[ss]) + crr * log(r[-1]) +
	                (1 - crr) * crpi * (log(pinf[0]) - log(cpie)) +
	                (1 - crr) * cry * (log(y[0]) - log(yflex[0])) +
	                crdy * (log(y[0]) - log(yflex[0]) - (log(y[-1]) - log(yflex[-1]))) +
	                log(ms[0]))

	afunc[0] = rk[ss] * 1 / cZcap * (exp(cZcap * (zcap[0] - 1)) - 1)

	afuncD[0] = rk[ss] * exp(cZcap * (zcap[0] - 1))

	Sfunc[0] = csadjcost / 2 * (cgamma * inve[0] / inve[-1] - cgamma) ^ 2

	SfuncD[0] = csadjcost * (cgamma * inve[0] / inve[-1] - cgamma)

	a[0] = 1 - crhoa + crhoa * a[-1] + z_ea / 100 * ea[x]

	b[0] = 1 - crhob + crhob * b[-1] +  z_eb / 100 * SCALE1_eb * eb[x]

	gy[0] - cg = crhog * (gy[-1] - cg) + z_eg / 100 * eg[x] + z_ea / 100 * ea[x] * cgy

	qs[0] = 1 - crhoqs + crhoqs * qs[-1] + z_eqs / 100 * SCALE1_eqs * eqs[x]

	ms[0] = 1 - crhoms + crhoms * ms[-1] + z_em / 100 * em[x]

	spinf[0] = 1 - crhopinf + crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]

	epinfma[0] = z_epinf / 100 * SCALE1_epinf * epinf[x]

	sw[0] = 1 - crhow + crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]

	ewma[0] = z_ew / 100 * SCALE1_ew * ew[x]

	# Flexible-price economy: pinf = cpie → Rotemberg cost vanishes
	yflex[0] = cflex[0] + inveflex[0] + gy[0] * yflex[ss] + afuncflex[0] * kpflex[-1] / cgamma

	yflex[0] = a[0] * kflex[0] ^ calfa * labflex[0] ^ (1 - calfa) - (cfc - 1) * yflex[ss]

	kflex[0] = kpflex[-1] * zcapflex[0] / cgamma

	kpflex[0] = inveflex[0] * qs[0] * (1 - Sfuncflex[0]) + kpflex[-1] * (1 - ctou) / cgamma

	xiflex[0] = exp((csigma - 1) / (1 + csigl) * labflex[0] ^ (1 + csigl)) * (cflex[0] - cflex[-1] * chabb / cgamma) ^ (-csigma)

	1 = qs[0] * pkflex[0] * (1 - Sfuncflex[0] - cgamma * inveflex[0] * SfuncDflex[0] / inveflex[-1]) + SfuncDflex[1] * qsaux[0] * xiflex[1] / xiflex[0] * pkflex[1] * (cgamma * inveflex[1] / inveflex[0]) ^ 2 * cbetabar

	xiflex[0] = xiflex[1] * b[0] * rrflex[0] * cbetabar

	rkflex[0] = afuncDflex[0]

	pkflex[0] = (rkflex[1] * zcapflex[1] - afuncflex[1] + (1 - ctou) * pkflex[1]) * xiflex[1] * cbetabar / xiflex[0]

	kflex[0] = labflex[0] * calfa / (1 - calfa) * wflex[0] / rkflex[0]

	mcflex = wflex[0] ^ (1 - calfa) * rkflex[0] ^ calfa / (a[0] * calfa ^ calfa * (1 - calfa) ^ (1 - calfa))

	wflex[0] * (1 + curvw * (1 - clandaw)) / (1 + curvW) = sw[ss] * (labflex[0] ^ csigl * clandaw * (cflex[0] - cflex[-1] * chabb / cgamma) + wflex[0] * curvW * (clandaw - 1) / (1 + curvW))

	afuncflex[0] = rkflex[ss] * 1 / cZcap * (exp(cZcap * (zcapflex[0] - 1)) - 1)

	afuncDflex[0] = rkflex[ss] * exp(cZcap * (zcapflex[0] - 1))

	Sfuncflex[0] = csadjcost / 2 * (cgamma * inveflex[0] / inveflex[-1] - cgamma) ^ 2

	SfuncDflex[0] = csadjcost * (cgamma * inveflex[0] / inveflex[-1] - cgamma)

	ygap[0] = 100 * log(y[0] / yflex[0])

	dy[0] = ctrend + 100 * (y[0] / y[-1] - 1)

	dc[0] = ctrend + 100 * (c[0] / c[-1] - 1)

	dinve[0] = ctrend + 100 * (inve[0] / inve[-1] - 1)

	pinfobs[0] = 100 * (pinf[0] - 1)

	robs[0] = 100 * (r[0] - 1)

	dwobs[0] = ctrend + 100 * (w[0] / w[-1] - 1)

	labobs[0] = constelab + 100 * (lab[0] / lab[ss] - 1)

end


@parameters Smets_Wouters_2007_HLT_obc_Rot simplify = false begin
	R_bar = 0.0 # log lower bound on gross rate (r >= 1.0)
	SCALE1_eb = -((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) ^ (-1)

	SCALE1_eqs = (cgamma ^ 2 * csadjcost) * (1 + cbeta * cgamma ^ (1 - csigma))

	# SCALE1_epinf: use baseline cprobp_base=0.6 for the markup shock normalization
	# (the Rotemberg model has no cprobp but needs an equivalent scale)
	SCALE1_epinf = 1 / ((1 / (1 + cbetabar * cgamma * cindp)) * ((1 - cprobp_base) * (1 - cbetabar * cgamma * cprobp_base) / cprobp_base) / ((cfc - 1) * curvp + 1))

	SCALE1_ew = 1 / ((1 - cprobw) * (1 - cbetabar * cgamma * cprobw) / ((1 + cbetabar * cgamma) * cprobw) * (1 / ((clandaw - 1) * curvw + 1)))

	cgamma 	= 1 + ctrend / 100          							# gross growth rate

	cbeta 	= 1 / (1 + constebeta / 100)    						# discount factor

	cZcap   = czcap / (1 - czcap)

    curvW = curvw * (1 - clandaw) / clandaw

	clandap = cfc                									# fixed cost share/gross price markup

	cbetabar= cbeta * cgamma ^ (-csigma)   							# growth-adjusted discount factor in Euler equation

    mcflex = mc[ss] | mcflex

    pinf[ss] = 1 + constepinf / 100 | cpie

	# Price demand elasticity (epsilon = cfc/(cfc-1), so epsilon-1 = 1/(cfc-1))
	ceps_p = cfc / (cfc - 1)

	# Rotemberg adjustment cost: calibrated to match baseline CES Calvo slope
	# Slope equivalence: (ceps_p-1)/phi_R = (1-xi)(1-beta*gamma*xi)/xi
	cprobp_base = 0.6  # baseline Calvo probability for calibration
	phi_R = (ceps_p - 1) * cprobp_base / ((1 - cprobp_base) * (1 - cbetabar * cgamma * cprobp_base))

	ctou = .025

	clandaw = 1.5

	cg = 0.18

	curvp = 10     # only used in SCALE1_epinf normalization

	curvw = 10

	calfa = .24

	csigma = 1.5

	cfc = 1.5

	cgy = 0.51

	csadjcost = 6.0144

	chabb = 0.6361

	cprobw = 0.8087

	csigl = 1.9423

	cindw = 0.3243

	cindp = 0.47

	czcap = 0.2696

	crpi = 1.488

	crr = 0.8762

	cry = 0.0593

	crdy = 0.2347

	crhoa = 0.9977

	crhob = 0.5799

	crhog = 0.9957

	crhoqs = 0.7165

	crhoms = 0

	crhopinf = 0

	crhow = 0

	cmap = 0

	cmaw = 0

	constelab = 0

	constepinf = 0.7

	constebeta = 0.7420

	ctrend = 0.3982

	z_ea	= 0.4618

	z_eb	= 1.8513

	z_eg	= 0.6090

	z_em	= 0.2397

	z_ew	= 0.2089

	z_eqs	= 0.6017

	z_epinf	= 0.1455

	1e-6 > ygap > -1e-6

	# SS solver hints
	mc > 0.3
	mc < 1.0
	y > 0.5
	c > 0.1
end
