using MacroModelling

@model NKV_EA_est begin
	Y_GAP[0] = Y_GAP[1] - 1 / sigma * (I[0] - PI[1]) - gamma_eta * ETA[0] - VX[0] * EPS_Y_GAP[x]

	ETA[0] = sigma_eta * EPS_ETA[x] + lambda_eta * ETA[-1] + lambda_eta_eta * AUX_ENDO_LAG_2_1[-1] - Y_GAP[0] * theta_y - Y_GAP[1] * theta_eta

	VX[0] = sigma_y - ETA[-1] * rho_eta1 - rho_eta2 * AUX_ENDO_LAG_2_1[-1] - rho_eta3 * AUX_ENDO_LAG_2_2[-1] - zeta_y * Y_GAP[-1]

	PI[0] = PI[1] * beta + Y_GAP[0] * kappa + sigma_pc * EPS_PI[x]

	I[0] = PI[0] * phi_pi + Y_GAP[0] * phi_y + sigma_I * EPS_MP[x]

	D_Y_GAP_PCT[0] = Y_GAP_PCT[0] - Y_GAP_PCT[-1]

	PI_OYA[0] = PI[0] + (PibarSS - 1) * 100

	Y_GAP_PCT[0] = Y_GAP[0] * 100

	D_PI_ANN_PCT[0] = PI_OYA[0] - PI_OYA[-1]

	ROBS[0] = I[0] * 4 + (Rbar - 1) * 400

	AUX_ENDO_LAG_2_1[0] = ETA[-1]

	AUX_ENDO_LAG_2_2[0] = AUX_ENDO_LAG_2_1[-1]

end


@parameters NKV_EA_est begin
	alpha = 0.3333333333333333

	beta = 0.99

	epsilon = 6

	phi = 1

	phi_pi = 1.5

	phi_y = 0.125

	sigma = 1

	theta = 0.6666666666666666

	gamma_eta = 0.01

	lambda_eta = 1.97

	lambda_eta_eta = (-1.01)

	sigma_y = 0.17

	theta_eta = 0.31

	theta_y = 0.08

	sigma_eta = 0.01

	sigma_I = 0.1

	sigma_pc = 0.1

	rho_eta1 = 0.75

	rho_eta2 = 0.25

	rho_eta3 = 0.25

	zeta_y = 0.75

	PibarSS = 1.00475

	Rbar = 1+PibarSS-1+1.003/beta-1

	omega = (1-alpha)/(1-alpha+alpha*epsilon)

	lambda = (1-theta)*(1-beta*theta)/theta*omega

	kappa = lambda*(sigma+(alpha+phi)/(1-alpha))

end

