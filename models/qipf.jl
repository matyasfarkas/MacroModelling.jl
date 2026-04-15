using MacroModelling

@model qipf begin
	LAM[0] = (C_TIL[0] - varkappa * C_TIL[-1] - SS_C_TIL * SS_NU) ^ (( - 1) / sigma) / (1 + SS_TAU_C)

	LAM[0] = beta * I[0] / PI_C[1] * LAM[1]

	W_TIL_C[0] ^ (1 + (1 + theta_w) / theta_w * chi) = (1 + theta_w) * Z_7[0] / Z_8[0] * z7_corr / z8_corr

	Z_7[0] = SS_VARSIGMA * chi_0 * W_C[0] ^ ((1 + theta_w) / theta_w * (1 + chi)) * N[0] ^ (1 + chi) / z7_corr + beta * xi_w * (PI_W[1] / PI_C[1]) ^ ((1 + chi) * ( - (1 + theta_w)) / theta_w) * Z_7[1]

	Z_8[0] = N[0] * LAM[0] * SS_VARSIGMA * (1 - SS_TAU_N) * SS_UPSILON_W * W_C[0] ^ ((1 + theta_w) / theta_w) / z8_corr + beta * xi_w * (PI_W[1] / PI_C[1]) ^ (( - 1) / theta_w) * Z_8[1]

	W_C[0] ^ (( - 1) / theta_w) = (1 - xi_w) * W_TIL_C[0] ^ (( - 1) / theta_w) + xi_w * (PI_W[0] / PI_C[0] * W_C[-1]) ^ (( - 1) / theta_w)

	PI_W[0] = PI_W[-1] ^ (1 - nu) * SS_PI_C ^ (nu * (1 - iota_w)) * (PI_C[-1] ^ (1 - iota_e) * (SS_PI_C_ST * PI_C[-1] * Q[-1] / AUX_ENDO_LAG_51_1[-1] / PI_C_ST[-1]) ^ iota_e) ^ (nu * iota_w)

	W_AMP_U[0] = (1 - xi_w) * (W_TIL_C[0] / W_C[0]) ^ ((1 + chi) * ( - (1 + theta_w)) / theta_w) + xi_w * W_AMP_U[-1] * (PI_W[0] / PI_C[0] * W_C[-1] / W_C[0]) ^ ((1 + chi) * ( - (1 + theta_w)) / theta_w)

	C_TIL[0] = C[0] + eta_0 * SS_G

	MC_D[0] = W_C[0] * 1 / (1 - alpha) * GAM_CD[0] * (N[0] / k) ^ alpha * 1 / SS_Z ^ (1 - alpha)

	Z_1[0] = Z_2[0] * P_TIL_D[0] - Z_3[0] * P_TIL_D[0] ^ (1 + (1 + theta_p) / theta_p * (1 + psi))

	Z_1[0] = MC_D[0] * LAM[0] * SS_VARSIGMA * (1 + theta_p) * (1 + psi) / (1 + psi + theta_p * psi) * Y_D[0] * VARTHETA[0] ^ ((1 + theta_p) / theta_p * (1 + psi)) / GAM_CD[0] + beta * xi_p * (PI_P[1] / PI_D[1]) ^ ((1 + psi) * ( - (1 + theta_p)) / theta_p) * Z_1[1]

	Z_2[0] = VARTHETA[0] ^ ((1 + theta_p) / theta_p * (1 + psi)) * Y_D[0] * LAM[0] * SS_VARSIGMA * SS_UPSILON / GAM_CD[0] + beta * xi_p * (PI_P[1] / PI_D[1]) ^ (( - (1 + psi + theta_p * psi)) / theta_p) * Z_2[1]

	Z_3[0] = Y_D[0] * SS_UPSILON * LAM[0] * SS_VARSIGMA * theta_p * psi / (1 + psi + theta_p * psi) / GAM_CD[0] + beta * xi_p * PI_P[1] / PI_D[1] * Z_3[1]

	PI_P[0] = SS_PI_D ^ (1 - iota) * PI_D[-1] ^ iota

	Y_D[0] * P_AMP_D[0] + P_AMP_M_ST[0] * Y_M_ST[0] = k ^ alpha * (N[0] * SS_Z) ^ (1 - alpha)

	P_AMP_D[0] = VARTHETA[0] ^ ((1 + theta_p) / theta_p * (1 + psi)) / (1 + psi) * Z_4[0] ^ ((1 + psi) * ( - (1 + theta_p)) / theta_p) + psi / (1 + psi)

	Z_4[0] ^ ((1 + psi) * ( - (1 + theta_p)) / theta_p) = (1 - xi_p) * P_TIL_D[0] ^ ((1 + psi) * ( - (1 + theta_p)) / theta_p) + xi_p * (PI_P[0] / PI_D[0] * Z_4[-1]) ^ ((1 + psi) * ( - (1 + theta_p)) / theta_p)

	P_AMP_M_ST[0] = VARTHETA_M[0] ^ ((1 + theta_p) / theta_p * (1 + psi_m)) / (1 + psi_m) * Z_M_4[0] ^ (( - (1 + theta_p)) / theta_p * (1 + psi_m)) + psi_m / (1 + psi_m)

	Z_M_4[0] ^ (( - (1 + theta_p)) * (1 + psi_m) / theta_p) = (1 - xi_m) * P_TIL_M_ST[0] ^ (( - (1 + theta_p)) * (1 + psi_m) / theta_p) + xi_m * (PI_PM_ST[0] / PI_M_ST[0] * Z_M_4[-1]) ^ (( - (1 + theta_p)) * (1 + psi_m) / theta_p)

	VARTHETA[0] = 1 + psi - psi * Z_5[0]

	Z_5[0] = P_TIL_D[0] * (1 - xi_p) + xi_p * PI_P[0] / PI_D[0] * Z_5[-1]

	VARTHETA[0] = Z_6[0]

	Z_6[0] ^ (( - (1 + psi + theta_p * psi)) / theta_p) = (1 - xi_p) * P_TIL_D[0] ^ (( - (1 + psi + theta_p * psi)) / theta_p) + xi_p * (PI_P[0] / PI_D[0] * Z_6[-1]) ^ (( - (1 + psi + theta_p * psi)) / theta_p)

	Y_D[0] = C[0] * (1 - omega_c) * GAM_CD[0] ^ ((1 + rho_c) / rho_c) + SS_G * (1 - omega_g) * GAM_GD[0] ^ ((1 + rho_g) / rho_g)

	Z_M_1[0] = P_TIL_M_ST[0] * Z_M_2[0] - Z_M_3[0] * P_TIL_M_ST[0] ^ (1 + (1 + theta_p) / theta_p * (1 + psi_m))

	Z_M_1[0] = MC_D[0] * VARTHETA_M[0] ^ ((1 + theta_p) / theta_p * (1 + psi_m)) * Y_M_ST[0] * LAM[0] * SS_VARSIGMA * (1 + theta_p) * (1 + psi_m) / (1 + psi_m + theta_p * psi_m) / GAM_CD[0] + beta * xi_m * (PI_PM_ST[1] / PI_M_ST[1]) ^ (( - (1 + theta_p)) / theta_p * (1 + psi_m)) * Z_M_1[1]

	Z_M_2[0] = VARTHETA_M[0] ^ ((1 + theta_p) / theta_p * (1 + psi_m)) * Y_M_ST[0] * LAM[0] * SS_VARSIGMA * SS_UPSILON_M_ST / GAM_CM_ST[0] * Q[0] + beta * xi_m * (PI_PM_ST[1] / PI_M_ST[1]) ^ (( - (1 + psi_m + theta_p * psi_m)) / theta_p) * Z_M_2[1]

	Z_M_3[0] = Q[0] * Y_M_ST[0] * SS_UPSILON_M_ST * LAM[0] * SS_VARSIGMA * theta_p * psi_m / (1 + psi_m + theta_p * psi_m) / GAM_CM_ST[0] + beta * xi_m * PI_PM_ST[1] / PI_M_ST[1] * Z_M_3[1]

	PI_PM_ST[0] = SS_PI_M_ST ^ (1 - iota_m) * PI_M_ST[-1] ^ iota_m

	VARTHETA_M[0] = 1 + psi_m - psi_m * Z_M_5[0]

	Z_M_5[0] = (1 - xi_m) * P_TIL_M_ST[0] + xi_m * PI_PM_ST[0] / PI_M_ST[0] * Z_M_5[-1]

	VARTHETA_M[0] = Z_M_6[0]

	Z_M_6[0] ^ (( - (1 + psi_m + theta_p * psi_m)) / theta_p) = (1 - xi_m) * P_TIL_M_ST[0] ^ (( - (1 + psi_m + theta_p * psi_m)) / theta_p) + xi_m * (PI_PM_ST[0] / PI_M_ST[0] * Z_M_6[-1]) ^ (( - (1 + psi_m + theta_p * psi_m)) / theta_p)

	Y_M_ST[0] = zeta_ST / zeta * (omega_c_ST * GAM_CM_ST[0] ^ ((1 + rho_c_ST) / rho_c_ST) * C_ST[0] + omega_g_ST * GAM_GM_ST[0] ^ ((1 + rho_g_ST) / rho_g_ST) * SS_G_ST)

	GAM_CD[0] = (1 - omega_c + omega_c * GAM_MD_TF[0] ^ (( - 1) / rho_c)) ^ (-rho_c)

	GAM_GD[0] = (1 - omega_g + omega_g * GAM_MD_TF[0] ^ (( - 1) / rho_g)) ^ (-rho_g)

	GAM_CM_ST[0] = (omega_c_ST + (1 - omega_c_ST) * GAM_MD_ST_TFST[0] ^ (1 / rho_c_ST)) ^ (-rho_c_ST)

	GAM_GM_ST[0] = (omega_g_ST + (1 - omega_g_ST) * GAM_MD_ST_TFST[0] ^ (1 / rho_g_ST)) ^ (-rho_g_ST)

	PI_C[0] = PI_D[0] * GAM_CD[0] / GAM_CD[-1]

	GAM_MD[0] / GAM_MD[-1] = PI_M[0] / PI_D[0]

	M_C[0] = C[0] * omega_c * GAM_CM[0] ^ ((1 + rho_c) / rho_c)

	M_G[0] = SS_G * omega_g * GAM_GM[0] ^ ((1 + rho_g) / rho_g)

	I[0] = (1 - psi_i) * (SS_I + psi_pi * (PI_C[0] - SS_PI_C) + psi_pid * (PI_D[0] - SS_PI_D)) + psi_i * I[-1] + SS_E_I #+psi_theta * THETA[0] 

	LAM_ST[0] = (C_TIL_ST[0] - varkappa_ST * C_TIL_ST[-1] - SS_C_TIL_ST * SS_NU_ST) ^ (( - 1) / sigma) / (1 + SS_TAU_C_ST)

	LAM_ST[0] = beta_ST * I_ST[0] / PI_C_ST[1] * LAM_ST[1]

	W_TIL_C_ST[0] ^ (1 + (1 + theta_w) / theta_w * chi) = (1 + theta_w) * Z_7_ST[0] / Z_8_ST[0] * z7_corr_ST / z8_corr_ST

	Z_7_ST[0] = SS_VARSIGMA_ST * chi_0_ST * W_C_ST[0] ^ ((1 + theta_w) / theta_w * (1 + chi)) * N_ST[0] ^ (1 + chi) / z7_corr_ST + beta * xi_w_ST * (PI_W_ST[1] / PI_C_ST[1]) ^ ((1 + chi) * ( - (1 + theta_w)) / theta_w) * Z_7_ST[1]

	Z_8_ST[0] = N_ST[0] * LAM_ST[0] * SS_VARSIGMA_ST * (1 - SS_TAU_N_ST) * SS_UPSILON_W_ST * W_C_ST[0] ^ ((1 + theta_w) / theta_w) / z8_corr_ST + beta * xi_w_ST * (PI_W_ST[1] / PI_C_ST[1]) ^ (( - 1) / theta_w) * Z_8_ST[1]

	W_C_ST[0] ^ (( - 1) / theta_w) = (1 - xi_w_ST) * W_TIL_C_ST[0] ^ (( - 1) / theta_w) + xi_w_ST * (PI_W_ST[0] / PI_C_ST[0] * W_C_ST[-1]) ^ (( - 1) / theta_w)

	PI_W_ST[0] = PI_W_ST[-1] ^ (1 - nu_ST) * SS_PI_C_ST ^ (nu_ST * (1 - iota_w_ST)) * PI_C_ST[-1] ^ (nu_ST * iota_w_ST)

	W_AMP_U_ST[0] = (1 - xi_w_ST) * (W_TIL_C_ST[0] / W_C_ST[0]) ^ ((1 + chi) * ( - (1 + theta_w)) / theta_w) + xi_w_ST * W_AMP_U_ST[-1] * (PI_W_ST[0] / PI_C_ST[0] * W_C_ST[-1] / W_C_ST[0]) ^ ((1 + chi) * ( - (1 + theta_w)) / theta_w)

	C_TIL_ST[0] = C_ST[0] + eta_0 * SS_G_ST

	MC_D_ST[0] = 1 / (1 - alpha) * W_C_ST[0] * GAM_CD_ST[0] * (N_ST[0] / k_ST) ^ alpha * 1 / SS_Z_ST ^ (1 - alpha)

	Z_1_ST[0] = Z_2_ST[0] * P_TIL_D_ST[0] - Z_3_ST[0] * P_TIL_D_ST[0] ^ (1 + (1 + theta_p) / theta_p * (1 + psi_ST))

	Z_1_ST[0] = MC_D_ST[0] * LAM_ST[0] * SS_VARSIGMA_ST * (1 + theta_p) * (1 + psi_ST) / (1 + psi_ST + theta_p * psi_ST) * Y_D_ST[0] * VARTHETA_ST[0] ^ ((1 + theta_p) / theta_p * (1 + psi_ST)) / GAM_CD_ST[0] + beta_ST * xi_p_ST * (PI_P_ST[1] / PI_D_ST[1]) ^ (( - (1 + theta_p)) / theta_p * (1 + psi_ST)) * Z_1_ST[1]

	Z_2_ST[0] = VARTHETA_ST[0] ^ ((1 + theta_p) / theta_p * (1 + psi_ST)) * Y_D_ST[0] * LAM_ST[0] * SS_VARSIGMA_ST * SS_UPSILON_ST / GAM_CD_ST[0] + beta_ST * xi_p_ST * (PI_P_ST[1] / PI_D_ST[1]) ^ (( - (1 + psi_ST + theta_p * psi_ST)) / theta_p) * Z_2_ST[1]

	Z_3_ST[0] = Y_D_ST[0] * SS_UPSILON_ST * LAM_ST[0] * SS_VARSIGMA_ST * theta_p * psi_ST / (1 + psi_ST + theta_p * psi_ST) / GAM_CD_ST[0] + beta_ST * xi_p_ST * PI_P_ST[1] / PI_D_ST[1] * Z_3_ST[1]

	PI_P_ST[0] = SS_PI_D_ST ^ (1 - iota_ST) * PI_D_ST[-1] ^ iota_ST

	Y_D_ST[0] * P_AMP_D_ST[0] + P_AMP_M[0] * Y_M[0] = k_ST ^ alpha * (N_ST[0] * SS_Z_ST) ^ (1 - alpha)

	P_AMP_D_ST[0] = VARTHETA_ST[0] ^ ((1 + theta_p) / theta_p * (1 + psi_ST)) / (1 + psi_ST) * Z_4_ST[0] ^ (( - (1 + theta_p)) / theta_p * (1 + psi_ST)) + psi_ST / (1 + psi_ST)

	Z_4_ST[0] ^ (( - (1 + theta_p)) * (1 + psi_ST) / theta_p) = (1 - xi_p_ST) * P_TIL_D_ST[0] ^ (( - (1 + theta_p)) * (1 + psi_ST) / theta_p) + xi_p_ST * (PI_P_ST[0] / PI_D_ST[0] * Z_4_ST[-1]) ^ (( - (1 + theta_p)) * (1 + psi_ST) / theta_p)

	P_AMP_M[0] = VARTHETA_M_ST[0] ^ ((1 + theta_p) / theta_p * (1 + psi_m_ST)) / (1 + psi_m_ST) * Z_M_4_ST[0] ^ (( - (1 + theta_p)) / theta_p * (1 + psi_m_ST)) + psi_m_ST / (1 + psi_m_ST)

	Z_M_4_ST[0] ^ (( - (1 + theta_p)) * (1 + psi_m_ST) / theta_p) = (1 - xi_m_ST) * P_TIL_M[0] ^ (( - (1 + theta_p)) * (1 + psi_m_ST) / theta_p) + xi_m_ST * (PI_PM[0] / PI_M[0] * Z_M_4_ST[-1]) ^ (( - (1 + theta_p)) * (1 + psi_m_ST) / theta_p)

	VARTHETA_ST[0] = 1 + psi_ST - psi_ST * Z_5_ST[0]

	Z_5_ST[0] = P_TIL_D_ST[0] * (1 - xi_p_ST) + xi_p_ST * PI_P_ST[0] / PI_D_ST[0] * Z_5_ST[-1]

	VARTHETA_ST[0] = Z_6_ST[0]

	Z_6_ST[0] ^ (( - (1 + psi_ST + theta_p * psi_ST)) / theta_p) = (1 - xi_p_ST) * P_TIL_D_ST[0] ^ (( - (1 + psi_ST + theta_p * psi_ST)) / theta_p) + xi_p_ST * (PI_P_ST[0] / PI_D_ST[0] * Z_6_ST[-1]) ^ (( - (1 + psi_ST + theta_p * psi_ST)) / theta_p)

	Y_D_ST[0] = C_ST[0] * (1 - omega_c_ST) * GAM_CD_ST[0] ^ ((1 + rho_c) / rho_c) + SS_G_ST * (1 - omega_g_ST) * GAM_GD_ST[0] ^ ((1 + rho_g) / rho_g)

	Z_M_1_ST[0] = P_TIL_M[0] * Z_M_2_ST[0] - Z_M_3_ST[0] * P_TIL_M[0] ^ (1 + (1 + theta_p) / theta_p * (1 + psi_m_ST))

	Z_M_1_ST[0] = MC_D_ST[0] * VARTHETA_M_ST[0] ^ ((1 + theta_p) / theta_p * (1 + psi_m_ST)) * Y_M[0] * LAM_ST[0] * SS_VARSIGMA_ST * (1 + theta_p) * (1 + psi_m_ST) / (1 + psi_m_ST + theta_p * psi_m_ST) / GAM_CD_ST[0] + beta_ST * xi_m_ST * (PI_PM[1] / PI_M[1]) ^ (( - (1 + theta_p)) / theta_p * (1 + psi_m_ST)) * Z_M_1_ST[1]

	Z_M_2_ST[0] = VARTHETA_M_ST[0] ^ ((1 + theta_p) / theta_p * (1 + psi_m_ST)) * Y_M[0] * LAM_ST[0] * SS_VARSIGMA_ST * SS_UPSILON_M / GAM_CM[0] / Q[0] + beta_ST * xi_m_ST * (PI_PM[1] / PI_M[1]) ^ (( - (1 + psi_m_ST + theta_p * psi_m_ST)) / theta_p) * Z_M_2_ST[1]

	Z_M_3_ST[0] = Y_M[0] * SS_UPSILON_M * LAM_ST[0] * SS_VARSIGMA_ST * theta_p * psi_m_ST / (1 + psi_m_ST + theta_p * psi_m_ST) / GAM_CM[0] / Q[0] + beta_ST * xi_m_ST * PI_PM[1] / PI_M[1] * Z_M_3_ST[1]

	PI_PM[0] = SS_PI_M ^ (1 - iota_m_ST) * PI_M[-1] ^ iota_m_ST

	VARTHETA_M_ST[0] = 1 + psi_m_ST - psi_m_ST * Z_M_5_ST[0]

	Z_M_5_ST[0] = (1 - xi_m_ST) * P_TIL_M[0] + xi_m_ST * PI_PM[0] / PI_M[0] * Z_M_5_ST[-1]

	VARTHETA_M_ST[0] = Z_M_6_ST[0]

	Z_M_6_ST[0] ^ (( - (1 + psi_m_ST + theta_p * psi_m_ST)) / theta_p) = (1 - xi_m_ST) * P_TIL_M[0] ^ (( - (1 + psi_m_ST + theta_p * psi_m_ST)) / theta_p) + xi_m_ST * (PI_PM[0] / PI_M[0] * Z_M_6_ST[-1]) ^ (( - (1 + psi_m_ST + theta_p * psi_m_ST)) / theta_p)

	Y_M[0] = zeta / zeta_ST * (C[0] * omega_c * GAM_CM[0] ^ ((1 + rho_c) / rho_c) + SS_G * omega_g * GAM_GM[0] ^ ((1 + rho_g) / rho_g))

	GAM_CD_ST[0] = (1 - omega_c_ST + omega_c_ST * GAM_MD_ST_TFST[0] ^ (( - 1) / rho_c_ST)) ^ (-rho_c_ST)

	GAM_GD_ST[0] = (1 - omega_g_ST + omega_g_ST * GAM_MD_ST_TFST[0] ^ (( - 1) / rho_g_ST)) ^ (-rho_g_ST)

	GAM_CM[0] = (omega_c + (1 - omega_c) * GAM_MD_TF[0] ^ (1 / rho_c)) ^ (-rho_c)

	GAM_GM[0] = (omega_g + (1 - omega_g) * GAM_MD_TF[0] ^ (1 / rho_g)) ^ (-rho_g)

	PI_C_ST[0] = PI_D_ST[0] * GAM_CD_ST[0] / GAM_CD_ST[-1]

	GAM_MD_ST[0] / GAM_MD_ST[-1] = PI_M_ST[0] / PI_D_ST[0]

	M_C_ST[0] = omega_c_ST * GAM_CM_ST[0] ^ ((1 + rho_c_ST) / rho_c_ST) * C_ST[0]

	M_G_ST[0] = omega_g_ST * GAM_GM_ST[0] ^ ((1 + rho_g_ST) / rho_g_ST) * SS_G_ST

	I_ST[0] = (1 - psi_i_ST) * (SS_I_ST + psi_pi_ST * (PI_C_ST[0] - SS_PI_C_ST) + psi_pid_ST * (PI_D_ST[0] - SS_PI_D_ST)) + psi_i_ST * I_ST[-1] + SS_E_I_ST

	I[0] * (1 - SS_TAU_F) = PI_C[1] * I_ST[0] * Q[1] / Q[0] / PI_C_ST[1] + I[0] * gamma_0 * var_e ^ gamma_1 * B_F[0] / (SS_Y_D + SS_Y_M_ST)

	B_F[0] = ( - B[0]) - SS_B_P + SS_B_M

	B[0] = Y_D[0] + (I[-1] * (1 - omega_f) / PI_D[0] + Q[0] * PI_C[0] * I_ST[-1] * omega_f / PI_D[0] / PI_C_ST[0] / Q[-1]) * B[-1] + (I[-1] / PI_D[0] - Q[0] * PI_C[0] * I_ST[-1] / PI_D[0] / PI_C_ST[0] / Q[-1]) * (SS_B_P * (omega_p - omega_f) - SS_B_M * (1 - omega_f)) + I[-1] * SS_TAU_F * (1 - cfm_nonfa) / PI_D[0] * ((1 - omega_f) * B_F[-1] + SS_B_P * (1 - omega_p)) + Y_M_ST[0] * Q[0] * GAM_CD[0] / GAM_CM_ST[0] - C[0] * GAM_CD[0] - SS_G * GAM_GD[0] #+ B[-1] * (1 - omega_b) * (IB[-1] - I[-1]) / PI_D[0] 

	# GAMMA[0] = gamma_0 * var_e ^ gamma_1

	#IB[0] = I[0] #+ THETA[0]

	# THETA[0] = 0

	BLIM[0] = B[0] + m * SS_Y

	Y[0] = Y_D[0] + Y_M_ST[0]

	Y_ST[0] = Y_D_ST[0] + Y_M[0]

	GAM_MD_TF[0] = GAM_MD[0] * exp(TF[0])

	GAM_MD_ST_TFST[0] = GAM_MD_ST[0] * exp(TF_ST[0])

	TF[0] = 0.9 * TF[-1] + EPS_TF[x]

	TF_ST[0] = TF[0]

	AUX_ENDO_LAG_51_1[0] = Q[-1]

end


@parameters qipf begin
	beta = 0.9963

	iota = 0.23

	iota_e = 0.0

	iota_m = 0.23

	iota_m_ST = 0.23

	iota_w = 0.5

	SS_PI_C = 1.005

	s_py = 0.8

	xi_m = 0.90

	xi_m_ST = 0.90

	xi_p = 0.92

	xi_w = 0.85

	alpha = 0.3

	beta_ST = 0.9963

	cfm_nonfa = 0

	chi = 1

	chi_0 = 1

	chi_0_ST = 1

	elb = 1

	elb_ST = 1

	eta_0 = 0

	gamma_0 = 0.06

	gamma_1 = 0

	iota_ST = 0.23

	iota_w_ST = 0.5

	nu = 1

	nu_ST = 1

	omega_c = 0.29

	omega_g = 0.1

	omega_f = 0.5

	omega_p = 0.5

	omega_b = 1

	psi = 0

	psi_ST = 0

	psi_m = 0

	psi_m_ST = 0

	psi_pi = 0

	psi_pi_ST = 0

	psi_pid = 1.5

	psi_pid_ST = 1.5

	psi_x = 0.0625

	psi_x_ST = 0.0625

	psi_i = 0

	psi_i_ST = 0

	psi_theta = 0

	rho_c = (-5)

	rho_c_ST = (-5)

	rho_g = (-5)

	rho_g_ST = (-5)

	rho_nu = 0.95

	rho_nu_ST = 0.95

	rho_tau_c = 0.95

	rho_tau_c_ST = 0.95

	rho_tau_f = 0.9

	rho_tau_n = 0.95

	rho_tau_n_ST = 0.95

	rho_varsigma = 0.95

	rho_varsigma_ST = 0.95

	rho_upsilon = 0.0

	rho_upsilon_ST = 0.0

	rho_upsilon_m = 0.0

	rho_upsilon_m_ST = 0.0

	rho_upsilon_w = 0.0

	rho_upsilon_w_ST = 0.0

	rho_e_i = 0.0

	rho_e_i_ST = 0.0

	rho_z = 0.95

	rho_z_ST = 0.95

	s_gy = 0.15

	s_gy_ST = 0.15

	s_my = 0.8

	sigma = 1

	spill_i = 0

	spill_upsilon_w = 0

	spill_varsigma = 0

	spill_z = 0

	theta_p = 0.2

	theta_w = 0.5

	tau_p = theta_p

	tau_w = theta_w

	var_e = 1

	varkappa = 0

	varkappa_ST = 0

	varrho_m = 0.95

	varrho_p = 0.95

	varrho_g = 0.967

	varrho_g_ST = 0.967

	xi_p_ST = 0.92

	xi_w_ST = 0.85

	zeta = 0.5

	zeta_ST = 1-zeta

	rho_tau_tf      = 0.9          

	SS_E_I = 0

	SS_E_I_ST = 0

	SS_GAM_CD = 1

	SS_GAM_CD_ST = 1

	SS_GAM_CM = 1

	SS_GAM_CM_ST = 1

	SS_GAM_GD = 1

	SS_GAM_GD_ST = 1

	SS_GAM_GM = 1

	SS_GAM_GM_ST = 1

	SS_GAM_MD = 1

	SS_GAM_MD_ST = 1

	SS_NU = 0.01

	SS_NU_ST = 0.01

	SS_P_AMP_D = 1

	SS_P_AMP_D_ST = 1

	SS_P_AMP_M = 1

	SS_P_AMP_M_ST = 1

	SS_P_TIL_D = 1

	SS_P_TIL_D_ST = 1

	SS_P_TIL_M = 1

	SS_P_TIL_M_ST = 1

	SS_PI_C_ST = 1.005

	SS_Q = 1

	SS_TAU_C = 0.15

	SS_TAU_C_ST = 0.15

	SS_TAU_F = 0

	SS_TAU_N = 0.15

	SS_TAU_N_ST = 0.15

	SS_UPSILON = 1+tau_p

	SS_UPSILON_ST = 1+tau_p

	SS_UPSILON_M_ST = 1+tau_p

	SS_UPSILON_M = 1+tau_p

	SS_UPSILON_W = 1+tau_w

	SS_UPSILON_W_ST = 1+tau_w

	SS_VARSIGMA = 1

	SS_VARSIGMA_ST = 1

	SS_VARTHETA = 1

	SS_VARTHETA_ST = 1

	SS_VARTHETA_M = 1

	SS_VARTHETA_M_ST = 1

	SS_W_AMP_U = 1

	SS_W_AMP_U_ST = 1

	SS_Z = 1

	SS_Z_ST = 1

	SS_Z_4 = 1

	SS_Z_4_ST = 1

	SS_Z_5 = 1

	SS_Z_5_ST = 1

	SS_Z_6 = 1

	SS_Z_6_ST = 1

	SS_Z_M_4 = 1

	SS_Z_M_4_ST = 1

	SS_Z_M_5 = 1

	SS_Z_M_5_ST = 1

	SS_Z_M_6 = 1

	SS_Z_M_6_ST = 1

	SS_GAMMA = gamma_0*var_e^gamma_1

	k_n = ((1+theta_p)/(1+tau_p)*(1-beta)/(beta*alpha))^(1/(alpha-1))

	SS_MC_D = (1+tau_p)/(1+theta_p)

	SS_MC_D_ST = (1+tau_p)/(1+theta_p)

	k_n_ST = ((1+theta_p)/(1+tau_p)*(1-beta_ST)/(alpha*beta_ST))^(1/(alpha-1))

	b_y = s_my-s_py-1/SS_GAMMA*(1-SS_TAU_F-beta/beta_ST)

	c_y = 1-s_gy-b_y*(1-(1-SS_TAU_F)*(1-omega_f)/beta-omega_f/beta_ST)-((1-SS_TAU_F)/beta-1/beta_ST)*(s_py*(omega_f-omega_p)+s_my*(1-omega_f))

	SS_N = (1/((c_y+s_gy*eta_0)*(1-varkappa-SS_NU))*((1+tau_p)*(1+tau_w)*(1-alpha)/(1+theta_p)*1/chi_0/(1+theta_w)*(1-SS_TAU_N)/(1+SS_TAU_C))^sigma*k_n^(alpha*(sigma-1)))^(1/(1+sigma*chi))

	k = k_n*SS_N

	SS_Y = SS_N*k_n^alpha

	SS_C = c_y*SS_Y

	SS_N_ST = QMIPF_solve_SS(zeta,zeta_ST,s_gy,s_gy_ST,SS_Y,SS_C,chi,sigma,eta_0,k_n_ST,alpha,theta_p,chi_0_ST,SS_TAU_N_ST,SS_TAU_C_ST,varkappa_ST,SS_NU_ST,SS_N,theta_w,tau_p,tau_w)

	k_ST = k_n_ST*SS_N_ST

	SS_Y_ST = SS_N_ST*k_n_ST^alpha

	SS_C_ST = zeta/zeta_ST*((1-s_gy)*SS_Y-SS_C)+SS_N_ST*k_n_ST^alpha*(1-s_gy_ST)

	SS_I = SS_PI_C/beta

	SS_I_ST = SS_PI_C_ST/beta_ST

	SS_IB = SS_I

	SS_THETA = 0

	SS_PI_D = SS_PI_C

	SS_PI_D_ST = SS_PI_C_ST

	SS_PI_M = SS_PI_C

	SS_PI_M_ST = SS_PI_C_ST

	SS_W_C = k_n^alpha*(1+tau_p)/(1+theta_p)*(1-alpha)

	SS_W_C_ST = k_n_ST^alpha*(1+tau_p)/(1+theta_p)*(1-alpha)

	SS_B = b_y*SS_Y

	SS_B_P = s_py*SS_Y

	SS_B_M = s_my*SS_Y

	SS_B_F = (-SS_B)-SS_B_P+SS_B_M

	SS_G = s_gy*SS_Y

	SS_G_ST = s_gy_ST*SS_Y_ST

	SS_C_TIL = SS_C+eta_0*SS_G

	SS_C_TIL_ST = SS_C_ST+eta_0*SS_G_ST

	SS_LAM = (SS_C_TIL-varkappa*SS_C_TIL-SS_NU*SS_C_TIL)^((-1)/sigma)/(1+SS_TAU_C)

	SS_LAM_ST = (SS_C_TIL_ST-varkappa_ST*SS_C_TIL_ST-SS_NU_ST*SS_C_TIL_ST)^((-1)/sigma)/(1+SS_TAU_C_ST)

	SS_W_TIL_C = SS_W_C

	SS_W_TIL_C_ST = SS_W_C_ST

	SS_PI_W = SS_PI_C

	SS_PI_W_ST = SS_PI_C_ST

	z7_corr = chi_0*SS_W_C^((1+theta_w)/theta_w*(1+chi))*SS_N^(1+chi)

	z8_corr = SS_N*(1-SS_TAU_N)*SS_LAM*SS_W_C^((1+theta_w)/theta_w)

	z7_corr_ST = chi_0_ST*SS_W_C_ST^((1+theta_w)/theta_w*(1+chi))*SS_N_ST^(1+chi)

	z8_corr_ST = SS_N_ST*SS_LAM_ST*(1-SS_TAU_N_ST)*SS_W_C_ST^((1+theta_w)/theta_w)

	SS_Z_7 = SS_N^(1+chi)*SS_W_C^((1+theta_w)/theta_w*(1+chi))*chi_0*SS_VARSIGMA/(1-beta*xi_w)/z7_corr

	SS_Z_7_ST = SS_N_ST^(1+chi)*SS_W_C_ST^((1+theta_w)/theta_w*(1+chi))*chi_0_ST*SS_VARSIGMA_ST/(1-beta*xi_w_ST)/z7_corr_ST

	SS_Z_8 = SS_N*SS_W_C^((1+theta_w)/theta_w)*(1+tau_w)*(1-SS_TAU_N)*SS_LAM*SS_VARSIGMA/(1-beta*xi_w)/z8_corr

	SS_Z_8_ST = SS_N_ST*SS_W_C_ST^((1+theta_w)/theta_w)*(1+tau_w)*(1-SS_TAU_N_ST)*SS_LAM_ST*SS_VARSIGMA_ST/(1-beta*xi_w_ST)/z8_corr_ST

	SS_PI_P = SS_PI_D

	SS_PI_P_ST = SS_PI_D_ST

	SS_PI_PM = SS_PI_D

	SS_PI_PM_ST = SS_PI_D_ST

	# SS_U = (1/(1-1/sigma)*(SS_C_TIL-varkappa*SS_C_TIL-SS_NU*SS_C_TIL)^(1-1/sigma)-SS_N^(1+chi)*chi_0*SS_W_AMP_U/(1+chi))/(1-beta)
 
	# SS_U_ST = (1/(1-1/sigma)*(SS_C_TIL_ST-varkappa_ST*SS_C_TIL_ST-SS_NU_ST*SS_C_TIL_ST)^(1-1/sigma)-SS_N_ST^(1+chi)*chi_0_ST*SS_W_AMP_U_ST/(1+chi))/(1-beta_ST)

	# if sigma==1

	SS_U = (log(SS_C_TIL-varkappa*SS_C_TIL-SS_NU*SS_C_TIL)-SS_N^(1+chi)*chi_0*SS_W_AMP_U/(1+chi))/(1-beta)

	SS_U_ST = (log(SS_C_TIL_ST-varkappa_ST*SS_C_TIL_ST-SS_NU_ST*SS_C_TIL_ST)-SS_N_ST^(1+chi)*chi_0_ST*SS_W_AMP_U_ST/(1+chi))/(1-beta_ST)

	# end
	s_omega_g_c_ST = 0.3448275862068966

	omega_c_ST = zeta/zeta_ST*(SS_Y-SS_C*(1-omega_c)-SS_G*(1-omega_g))/(SS_C_ST+SS_G_ST*s_omega_g_c_ST)

	omega_g_ST = s_omega_g_c_ST*omega_c_ST

	SS_Y_M = zeta/zeta_ST*(SS_C*omega_c+SS_G*omega_g)

	SS_Y_M_ST = zeta_ST/zeta*(SS_C_ST*omega_c_ST+SS_G_ST*omega_g_ST)

	SS_Y_D = SS_Y-SS_Y_M_ST

	SS_Y_D_ST = SS_Y_ST-SS_Y_M

	SS_Z_1 = SS_LAM*(1+theta_p)*(1+psi)/((1-beta*xi_p)*(1+psi+theta_p*psi))*SS_Y_D*SS_MC_D

	SS_Z_1_ST = SS_LAM_ST*(1+theta_p)*(1+psi_ST)/((1-beta_ST*xi_p_ST)*(1+psi_ST+theta_p*psi_ST))*SS_Y_D_ST*SS_MC_D_ST

	SS_Z_2 = SS_Y_D*(1+tau_p)*SS_LAM/(1-beta*xi_p)

	SS_Z_2_ST = SS_Y_D_ST*(1+tau_p)*SS_LAM_ST/(1-beta_ST*xi_p_ST)

	SS_Z_3 = SS_Y_D*SS_LAM*(1+tau_p)*theta_p*psi/(1+psi+theta_p*psi)/(1-beta*xi_p)

	SS_Z_3_ST = SS_Y_D_ST*SS_LAM_ST*(1+tau_p)*theta_p*psi_ST/(1+psi_ST+theta_p*psi_ST)/(1-beta_ST*xi_p_ST)

	SS_Z_M_1 = SS_MC_D*SS_Y_M_ST*SS_LAM*(1+theta_p)*(1+psi_m)/((1-beta*xi_m)*(1+psi_m+theta_p*psi_m))

	SS_Z_M_1_ST = SS_MC_D_ST*SS_Y_M*SS_LAM_ST*(1+theta_p)*(1+psi_m_ST)/((1-beta_ST*xi_m_ST)*(1+psi_m_ST+theta_p*psi_m_ST))

	SS_Z_M_2 = (1+tau_p)*SS_LAM*SS_Y_M_ST/(1-beta*xi_m)

	SS_Z_M_2_ST = (1+tau_p)*SS_LAM_ST*SS_Y_M/(1-beta_ST*xi_m_ST)

	SS_Z_M_3 = (1+tau_p)*SS_Y_M_ST*SS_LAM*theta_p*psi_m/(1+psi_m+theta_p*psi_m)/(1-beta*xi_m)

	SS_Z_M_3_ST = (1+tau_p)*SS_Y_M*SS_LAM_ST*theta_p*psi_m_ST/(1+psi_m_ST+theta_p*psi_m_ST)/(1-beta_ST*xi_m_ST)

	SS_M_C = SS_C*omega_c

	SS_M_C_ST = SS_C_ST*omega_c_ST

	SS_M_G = SS_G*omega_g

	SS_M_G_ST = SS_G_ST*omega_g_ST

	m = (-SS_B)/SS_Y+0.48

	SS_BLIM = SS_B+SS_Y*m

	
end
