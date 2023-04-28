using MacroModelling

@model NKV begin
	Y_GAP[0] = Y_GAP[1] - 1 / sigma * (I[0] - PI[1]) - gamma_eta * ETA[0] - sigma_y * EPS_Y_GAP[x]

	ETA[0] = lambda_eta * ETA[-1] + lambda_eta_eta * AUX_ENDO_LAG_2_1[-1] - Y_GAP[0] * theta_y - Y_GAP[1] * theta_eta

	PI[0] = PI[1] * beta + Y_GAP[0] * kappa

	I[0] = PI[0] * phi_pi + Y_GAP[0] * phi_y

	D_Y_GAP_PCT[0] = Y_GAP_PCT[0] - Y_GAP_PCT[-1]

	PI_OYA[0] = 100 * (PI[0] + PI[-1] + AUX_ENDO_LAG_4_1[-1] + AUX_ENDO_LAG_4_2[-1])

	Y_GAP_PCT[0] = Y_GAP[0] * 100

	D_PI_ANN_PCT[0] = PI_OYA[0] - PI_OYA[-1]

	AUX_ENDO_LAG_2_1[0] = ETA[-1]

	AUX_ENDO_LAG_4_1[0] = PI[-1]

	AUX_ENDO_LAG_4_2[0] = AUX_ENDO_LAG_4_1[-1]

end


@parameters NKV begin
	alpha = 0.3333333333333333

	beta = 0.99

	epsilon = 6

	phi = 1

	phi_pi = 1.5

	phi_y = 0.125

	sigma = 1

	theta = 0.6666666666666666

	gamma_eta = (-534.96)

	lambda_eta = (-6.49)

	lambda_eta_eta = (-18.55)

	sigma_y = 16.48

	theta_eta = (-4.38)

	theta_y = 14.46

	omega = (1-alpha)/(1-alpha+alpha*epsilon)

	lambda = (1-theta)*(1-theta*beta)/theta*omega

	kappa = lambda*(sigma+(alpha+phi)/(1-alpha))

end

