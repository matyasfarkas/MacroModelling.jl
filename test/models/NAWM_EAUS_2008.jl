@model NAWM_EAUS_2008 begin
	EA_R[0] ^ 4 - 1 = EA_PHIRR * (EA_R[-1] ^ 4 - 1) + (1 - EA_PHIRR) * (EA_RRSTAR ^ 4 * EA_PI4TARGET - 1 + EA_PHIRPI * (EA_PIC4[0] - EA_PI4TARGET)) + EA_PHIRGY * (EA_Y[0] / EA_Y[-1] - 1) + EA_EPSR[x]

	US_R[0] ^ 4 - 1 = US_PHIRR * (US_R[-1] ^ 4 - 1) + (1 - US_PHIRR) * (US_RRSTAR ^ 4 * US_PI4TARGET - 1 + US_PHIRPI * (US_PIC4[0] - US_PI4TARGET)) + US_PHIRGY * (US_Y[0] / US_Y[-1] - 1) + US_EPSR[x]

	EA_UTILI[0] = 1 / (1 - EA_SIGMA) * (EA_CI[0] - EA_KAPPA * EA_CI[-1]) ^ (1 - EA_SIGMA) - 1 / (1 + EA_ZETA) * EA_NI[0] ^ (1 + EA_ZETA) + EA_BETA * EA_UTILI[1]

	EA_LAMBDAI[0] * (1 + EA_TAUC[0] + EA_GAMMAVI[0] + EA_VI[0] * EA_GAMMAVIDER[0]) = (EA_CI[0] - EA_KAPPA * EA_CI[-1]) ^ (-EA_SIGMA)

	EA_R[0] = EA_LAMBDAI[0] * EA_BETA ^ (-1) / EA_LAMBDAI[1] * EA_PIC[1]

	EA_GAMMAVIDER[0] * EA_VI[0] ^ 2 = 1 - EA_BETA * EA_LAMBDAI[1] / (EA_LAMBDAI[0] * EA_PIC[1])

	EA_VI[0] = EA_CI[0] * (1 + EA_TAUC[0]) / EA_MI[0]

	EA_GAMMAVI[0] = EA_VI[0] * EA_GAMMAV1 + EA_GAMMAV2 / EA_VI[0] - 2 * (EA_GAMMAV1 * EA_GAMMAV2) ^ 0.5

	EA_GAMMAVIDER[0] = EA_GAMMAV1 - EA_GAMMAV2 * EA_VI[0] ^ (-2)

	EA_KI[0] = (1 - EA_DELTA) * EA_KI[-1] + (1 - EA_GAMMAI[-1]) * EA_II[-1]

	EA_GAMMAI[0] = EA_GAMMAI1 / 2 * (EA_II[0] / EA_II[-1] - 1) ^ 2

	EA_GAMMAIDER[0] = EA_GAMMAI1 * (EA_II[0] / EA_II[-1] - 1) / EA_II[-1]

	EA_GAMMAU[0] = ((EA_DELTA + EA_BETA ^ (-1) - 1) * EA_QBAR - EA_DELTA * EA_TAUKBAR * EA_PIBAR) / (EA_PIBAR * (1 - EA_TAUKBAR)) * (EA_U[0] - 1) + EA_GAMMAU2 / 2 * (EA_U[0] - 1) ^ 2

	EA_GAMMAUDER[0] = ((EA_DELTA + EA_BETA ^ (-1) - 1) * EA_QBAR - EA_DELTA * EA_TAUKBAR * EA_PIBAR) / (EA_PIBAR * (1 - EA_TAUKBAR)) + (EA_U[0] - 1) * EA_GAMMAU2

	EA_RK[0] = EA_GAMMAUDER[0] * EA_PI[0]

	EA_PI[0] = EA_Q[0] * (1 - EA_GAMMAI[0] - EA_II[0] * EA_GAMMAIDER[0]) + EA_BETA * EA_LAMBDAI[1] / EA_LAMBDAI[0] * EA_Q[1] * EA_GAMMAIDER[1] * EA_II[1] ^ 2 / EA_II[0]

	EA_Q[0] = EA_BETA * EA_LAMBDAI[1] / EA_LAMBDAI[0] * ((1 - EA_TAUK[1]) * (EA_RK[1] * EA_U[1] - EA_GAMMAU[1] * EA_PI[1]) + EA_PI[1] * EA_DELTA * EA_TAUK[1] + (1 - EA_DELTA) * EA_Q[1])

	EA_WITILDE[0] ^ (1 + EA_ZETA * EA_ETAI) = EA_ETAI / (EA_ETAI - 1) * EA_FI[0] / EA_GI[0]

	EA_FI[0] = EA_WI[0] ^ ((1 + EA_ZETA) * EA_ETAI) * EA_NDI[0] ^ (1 + EA_ZETA) + EA_BETA * EA_XII * (EA_PIC[1] / (EA_PIC[0] ^ EA_CHII * EA_PI4TARGET ^ (0.25 * (1 - EA_CHII)))) ^ ((1 + EA_ZETA) * EA_ETAI) * EA_FI[1]

	EA_GI[0] = EA_NDI[0] * EA_LAMBDAI[0] * (1 - EA_TAUN[0] - EA_TAUWH[0]) * EA_WI[0] ^ EA_ETAI + EA_BETA * EA_XII * (EA_PIC[1] / (EA_PIC[0] ^ EA_CHII * EA_PI4TARGET ^ (0.25 * (1 - EA_CHII)))) ^ (EA_ETAI - 1) * EA_GI[1]

	EA_WI[0] ^ (1 - EA_ETAI) = (1 - EA_XII) * EA_WITILDE[0] ^ (1 - EA_ETAI) + EA_XII * EA_WI[-1] ^ (1 - EA_ETAI) * (EA_PI4TARGET ^ (0.25 * (1 - EA_CHII)) * EA_PIC[-1] ^ EA_CHII / EA_PIC[0]) ^ (1 - EA_ETAI)

	EA_UTILJ[0] = 1 / (1 - EA_SIGMA) * (EA_CJ[0] - EA_KAPPA * EA_CJ[-1]) ^ (1 - EA_SIGMA) - 1 / (1 + EA_ZETA) * EA_NJ[0] ^ (1 + EA_ZETA) + EA_BETA * EA_UTILJ[1]

	EA_CJ[0] * (1 + EA_TAUC[0] + EA_GAMMAVJ[0]) + EA_MJ[0] = EA_NJ[0] * (1 - EA_TAUN[0] - EA_TAUWH[0]) * EA_WJ[0] + EA_TRJ[0] - EA_TJ[0] + EA_MJ[-1] * EA_PIC[0] ^ (-1)

	EA_LAMBDAJ[0] * (1 + EA_TAUC[0] + EA_GAMMAVJ[0] + EA_VJ[0] * EA_GAMMAVJDER[0]) = (EA_CJ[0] - EA_KAPPA * EA_CJ[-1]) ^ (-EA_SIGMA)

	EA_GAMMAVJDER[0] * EA_VJ[0] ^ 2 = 1 - EA_BETA * EA_LAMBDAJ[1] / (EA_PIC[1] * EA_LAMBDAJ[0])

	EA_VJ[0] = (1 + EA_TAUC[0]) * EA_CJ[0] / EA_MJ[0]

	EA_GAMMAVJ[0] = EA_GAMMAV1 * EA_VJ[0] + EA_GAMMAV2 / EA_VJ[0] - 2 * (EA_GAMMAV1 * EA_GAMMAV2) ^ 0.5

	EA_GAMMAVJDER[0] = EA_GAMMAV1 - EA_GAMMAV2 * EA_VJ[0] ^ (-2)

	EA_WJTILDE[0] ^ (1 + EA_ZETA * EA_ETAJ) = EA_ETAJ / (EA_ETAJ - 1) * EA_FJ[0] / EA_GJ[0]

	EA_FJ[0] = EA_WJ[0] ^ ((1 + EA_ZETA) * EA_ETAJ) * EA_NDJ[0] ^ (1 + EA_ZETA) + EA_BETA * EA_XIJ * (EA_PIC[1] / (EA_PIC[0] ^ EA_CHIJ * EA_PI4TARGET ^ (0.25 * (1 - EA_CHIJ)))) ^ ((1 + EA_ZETA) * EA_ETAJ) * EA_FJ[1]

	EA_GJ[0] = EA_NDJ[0] * (1 - EA_TAUN[0] - EA_TAUWH[0]) * EA_LAMBDAJ[0] * EA_WJ[0] ^ EA_ETAJ + EA_BETA * EA_XIJ * (EA_PIC[1] / (EA_PIC[0] ^ EA_CHIJ * EA_PI4TARGET ^ (0.25 * (1 - EA_CHIJ)))) ^ (EA_ETAJ - 1) * EA_GJ[1]

	EA_WJ[0] ^ (1 - EA_ETAJ) = (1 - EA_XIJ) * EA_WJTILDE[0] ^ (1 - EA_ETAJ) + EA_XIJ * EA_WJ[-1] ^ (1 - EA_ETAJ) * (EA_PI4TARGET ^ (0.25 * (1 - EA_CHIJ)) * EA_PIC[-1] ^ EA_CHIJ / EA_PIC[0]) ^ (1 - EA_ETAJ)

	EA_YS[0] = EA_Z[0] * EA_KD[0] ^ EA_ALPHA * EA_ND[0] ^ (1 - EA_ALPHA) - EA_PSIBAR

	EA_RK[0] = EA_ALPHA * (EA_YS[0] + EA_PSIBAR) / EA_KD[0] * EA_MC[0]

	EA_MC[0] = 1 / (EA_Z[0] * EA_ALPHA ^ EA_ALPHA * (1 - EA_ALPHA) ^ (1 - EA_ALPHA)) * EA_RK[0] ^ EA_ALPHA * ((1 + EA_TAUWF[0]) * EA_W[0]) ^ (1 - EA_ALPHA)

	EA_NDI[0] = EA_ND[0] * (1 - EA_OMEGA) * (EA_WI[0] / EA_W[0]) ^ (-EA_ETA)

	EA_NDJ[0] = EA_ND[0] * EA_OMEGA * (EA_WJ[0] / EA_W[0]) ^ (-EA_ETA)

	EA_ND[0] ^ (1 - 1 / EA_ETA) = (1 - EA_OMEGA) ^ (1 / EA_ETA) * EA_NDI[0] ^ (1 - 1 / EA_ETA) + EA_OMEGA ^ (1 / EA_ETA) * EA_NDJ[0] ^ (1 - 1 / EA_ETA)

	EA_D[0] = EA_Y[0] * EA_PY[0] - EA_RK[0] * EA_KD[0] - EA_ND[0] * (1 + EA_TAUWF[0]) * EA_W[0]

	EA_PHTILDE[0] / EA_PH[0] = EA_THETA / (EA_THETA - 1) * EA_FH[0] / EA_GH[0]

	EA_FH[0] = EA_MC[0] * EA_H[0] + EA_LAMBDAI[1] * EA_BETA * EA_XIH / EA_LAMBDAI[0] * (EA_PIH[1] / (EA_PIH[0] ^ EA_CHIH * EA_PI4TARGET ^ (0.25 * (1 - EA_CHIH)))) ^ EA_THETA * EA_FH[1]

	EA_GH[0] = EA_PH[0] * EA_H[0] + EA_LAMBDAI[1] * EA_BETA * EA_XIH / EA_LAMBDAI[0] * (EA_PIH[1] / (EA_PIH[0] ^ EA_CHIH * EA_PI4TARGET ^ (0.25 * (1 - EA_CHIH)))) ^ (EA_THETA - 1) * EA_GH[1]

	EA_PH[0] ^ (1 - EA_THETA) = (1 - EA_XIH) * EA_PHTILDE[0] ^ (1 - EA_THETA) + EA_XIH * (EA_PH[-1] / EA_PIC[0]) ^ (1 - EA_THETA) * (EA_PI4TARGET ^ (0.25 * (1 - EA_CHIH)) * EA_PIH[-1] ^ EA_CHIH) ^ (1 - EA_THETA)

	EA_PIH[0] = EA_PIC[0] * EA_PH[0] / EA_PH[-1]

	US_PIMTILDE[0] / US_PIM[0] = EA_THETA / (EA_THETA - 1) * EA_FX[0] / EA_GX[0]

	EA_FX[0] = EA_MC[0] * US_SIZE / EA_SIZE * US_IM[0] + EA_LAMBDAI[1] * EA_BETA * EA_XIX / EA_LAMBDAI[0] * (US_PIIM[1] / (US_PIIM[0] ^ EA_CHIX * EA_PI4TARGET ^ (0.25 * (1 - EA_CHIX)))) ^ EA_THETA * EA_FX[1]

	EA_GX[0] = US_IM[0] * US_SIZE * US_PIM[0] * EAUS_RER[0] / EA_SIZE + EA_LAMBDAI[1] * EA_BETA * EA_XIX / EA_LAMBDAI[0] * (US_PIIM[1] / (US_PIIM[0] ^ EA_CHIX * EA_PI4TARGET ^ (0.25 * (1 - EA_CHIX)))) ^ (EA_THETA - 1) * EA_GX[1]

	US_PIM[0] ^ (1 - EA_THETA) = (1 - EA_XIX) * US_PIMTILDE[0] ^ (1 - EA_THETA) + EA_XIX * (US_PIM[-1] / US_PIC[0]) ^ (1 - EA_THETA) * (US_PIIM[-1] ^ EA_CHIX * US_PI4TARGET ^ (0.25 * (1 - EA_CHIH))) ^ (1 - EA_THETA)

	US_PIIM[0] = US_PIC[0] * US_PIM[0] / US_PIM[-1]

	EAUS_RER[0] = EA_RER[0] / US_RER

	EA_QC[0] ^ ((EA_MUC - 1) / EA_MUC) = EA_NUC ^ (1 / EA_MUC) * EA_HC[0] ^ (1 - 1 / EA_MUC) + (1 - EA_NUC) ^ (1 / EA_MUC) * ((1 - EA_GAMMAIMC[0]) * EA_IMC[0]) ^ (1 - 1 / EA_MUC)

	1 = EA_NUC * EA_PH[0] ^ (1 - EA_MUC) + (1 - EA_NUC) * (EA_PIM[0] / EA_GAMMAIMCDAG[0]) ^ (1 - EA_MUC)

	EA_HC[0] = EA_QC[0] * EA_NUC * EA_PH[0] ^ (-EA_MUC)

	EA_GAMMAIMC[0] = EA_GAMMAIMC1 / 2 * (EA_IMC[0] / EA_QC[0] / (EA_IMC[-1] / EA_QC[-1]) - 1) ^ 2

	EA_GAMMAIMCDAG[0] = 1 - EA_GAMMAIMC[0] - EA_IMC[0] * EA_GAMMAIMC1 * (EA_IMC[0] / EA_QC[0] / (EA_IMC[-1] / EA_QC[-1]) - 1) / EA_QC[0] / (EA_IMC[-1] / EA_QC[-1])

	EA_QI[0] ^ ((EA_MUI - 1) / EA_MUI) = EA_NUI ^ (1 / EA_MUI) * EA_HI[0] ^ (1 - 1 / EA_MUI) + (1 - EA_NUI) ^ (1 / EA_MUI) * ((1 - EA_GAMMAIMI[0]) * EA_IMI[0]) ^ (1 - 1 / EA_MUI)

	EA_PI[0] ^ (1 - EA_MUI) = EA_NUI * EA_PH[0] ^ (1 - EA_MUI) + (1 - EA_NUI) * (EA_PIM[0] / EA_GAMMAIMIDAG[0]) ^ (1 - EA_MUI)

	EA_HI[0] = EA_QI[0] * EA_NUI * (EA_PH[0] / EA_PI[0]) ^ (-EA_MUI)

	EA_GAMMAIMI[0] = EA_GAMMAIMI1 / 2 * (EA_IMI[0] / EA_QI[0] / (EA_IMI[-1] / EA_QI[-1]) - 1) ^ 2

	EA_GAMMAIMIDAG[0] = 1 - EA_GAMMAIMI[0] - EA_IMI[0] * EA_GAMMAIMI1 * (EA_IMI[0] / EA_QI[0] / (EA_IMI[-1] / EA_QI[0]) - 1) / EA_QI[0] / (EA_IMI[-1] / EA_QI[-1])

	EA_PH[-1] * EA_G[-1] + EA_TR[-1] + EA_B[-1] * EA_PIC[-1] ^ (-1) + EA_PIC[-1] ^ (-1) * EA_M[-2] = EA_TAUC[-1] * EA_C[-1] + (EA_TAUN[-1] + EA_TAUWH[-1]) * (EA_WI[-1] * EA_NDI[-1] + EA_WJ[-1] * EA_NDJ[-1]) + EA_TAUWF[-1] * EA_W[-1] * EA_ND[-1] + EA_TAUK[-1] * (EA_RK[-1] * EA_U[-1] - (EA_DELTA + EA_GAMMAU[-1]) * EA_PI[-1]) * EA_K[-1] + EA_TAUD[-1] * EA_D[-1] + EA_T[-1] + EA_R[-1] ^ (-1) * EA_B[0] + EA_M[-1]

	EA_PH[0] * EA_G[0] = EA_GY[0] * EA_PYBAR * EA_YBAR

	EA_TR[0] = EA_YBAR * EA_PYBAR * EA_TRY[0]

	EA_T[0] / (EA_PYBAR * EA_YBAR) = EA_PHITB * (EA_B[0] / (EA_PYBAR * EA_YBAR) - EA_BYTARGET)

	EA_TI[0] = EA_T[0] * EA_UPSILONT

	EA_TRI[0] = EA_TR[0] * EA_UPSILONTR

	EA_PIC4[0] = EA_PIC[0] * EA_PIC[-1] * EA_PIC[-2] * EA_PIC[-3]

	EA_RR[0] - 1 = EA_R[0] / EA_PIC[1] - 1

	EA_C[0] = EA_CI[0] * (1 - EA_OMEGA) + EA_CJ[0] * EA_OMEGA

	EA_M[0] = EA_MI[0] * (1 - EA_OMEGA) + EA_MJ[0] * EA_OMEGA

	EA_K[0] = EA_KI[0] * (1 - EA_OMEGA)

	EA_I[0] = EA_II[0] * (1 - EA_OMEGA)

	EA_TRJ[0] = EA_TR[0] * 1 / EA_OMEGA - EA_TRI[0] * (1 - EA_OMEGA) / EA_OMEGA

	EA_TJ[0] = EA_T[0] * 1 / EA_OMEGA - EA_TI[0] * (1 - EA_OMEGA) / EA_OMEGA

	EA_GAMMAV[0] = EA_GAMMAVI[0] * EA_CI[0] * (1 - EA_OMEGA) + EA_GAMMAVJ[0] * EA_CJ[0] * EA_OMEGA

	EA_NI[0] = EA_NDI[0] * EA_SI[0]

	EA_SI[0] = (1 - EA_XII) * (EA_WITILDE[0] / EA_WI[0]) ^ (-EA_ETAI) + EA_XII * (EA_WI[-1] / EA_WI[0]) ^ (-EA_ETAI) * (EA_PIC[0] / (EA_PI4TARGET ^ (0.25 * (1 - EA_CHII)) * EA_PIC[-1] ^ EA_CHII)) ^ EA_ETAI * EA_SI[-1]

	EA_NJ[0] = EA_NDJ[0] * EA_SJ[0]

	EA_SJ[0] = (1 - EA_XIJ) * (EA_WJTILDE[0] / EA_WJ[0]) ^ (-EA_ETAJ) + EA_XIJ * (EA_WJ[-1] / EA_WJ[0]) ^ (-EA_ETAJ) * (EA_PIC[0] / (EA_PI4TARGET ^ (0.25 * (1 - EA_CHIJ)) * EA_PIC[-1] ^ EA_CHIJ)) ^ EA_ETAJ * EA_SJ[-1]

	EA_U[0] * EA_K[0] = EA_KD[0]

	EA_YS[0] = EA_H[0] * EA_SH[0] + US_IM[0] * US_SIZE * EA_SX[0] / EA_SIZE

	EA_H[0] = EA_G[0] + EA_HC[0] + EA_HI[0]

	EA_IM[0] = EA_IMC[0] + EA_IMI[0]

	EA_SH[0] = (1 - EA_XIH) * (EA_PHTILDE[0] / EA_PH[0]) ^ (-EA_THETA) + EA_XIH * (EA_PIH[0] / (EA_PI4TARGET ^ (0.25 * (1 - EA_CHIH)) * EA_PIH[-1] ^ EA_CHIH)) ^ EA_THETA * EA_SH[-1]

	EA_SX[0] = (1 - EA_XIX) * (US_PIMTILDE[0] / US_PIM[0]) ^ (-EA_THETA) + EA_XIX * (US_PIIM[0] / (EA_PI4TARGET ^ (0.25 * (1 - EA_CHIH)) * US_PIIM[-1] ^ EA_CHIX)) ^ EA_THETA * EA_SX[-1]

	EA_QC[0] = EA_C[0] + EA_GAMMAV[0]

	EA_QI[0] = EA_I[0] + EA_GAMMAU[0] * EA_K[0]

	EA_Y[0] * EA_PY[0] = US_IM[0] * US_SIZE * US_PIM[0] * EAUS_RER[0] / EA_SIZE + EA_PH[0] * EA_G[0] + EA_QC[0] + EA_PI[0] * EA_QI[0] - EA_PIM[0] * ((1 - EA_GAMMAIMC[0]) * EA_IMC[0] / EA_GAMMAIMCDAG[0] + (1 - EA_GAMMAIMI[0]) * EA_IMI[0] / EA_GAMMAIMIDAG[0])

	EA_Y[0] = EA_YS[0]

	log(EA_Z[0]) = (1 - EA_RHOZ) * log(EA_ZBAR) + EA_RHOZ * log(EA_Z[-1]) + EA_EPSZ[x]

	EA_GY[0] = (1 - EA_RHOG) * EA_GYBAR + EA_RHOG * EA_GY[-1] + EA_EPSG[x]

	EA_TRY[0] = (1 - EA_RHOTR) * EA_TRYBAR + EA_RHOTR * EA_TRY[-1] + EA_EPSTR[x]

	EA_TAUC[0] = (1 - EA_RHOTAUC) * EA_TAUCBAR + EA_TAUC[-1] * EA_RHOTAUC + EA_EPSTAUC[x]

	EA_TAUD[0] = (1 - EA_RHOTAUD) * EA_TAUDBAR + EA_TAUD[-1] * EA_RHOTAUD + EA_EPSTAUD[x]

	EA_TAUK[0] = EA_TAUKBAR * (1 - EA_RHOTAUK) + EA_TAUK[-1] * EA_RHOTAUK + EA_EPSTAUK[x]

	EA_TAUN[0] = (1 - EA_RHOTAUN) * EA_TAUNBAR + EA_TAUN[-1] * EA_RHOTAUN + EA_EPSTAUN[x]

	EA_TAUWH[0] = (1 - EA_RHOTAUWH) * EA_TAUWHBAR + EA_TAUWH[-1] * EA_RHOTAUWH + EA_EPSTAUWH[x]

	EA_TAUWF[0] = (1 - EA_RHOTAUWF) * EA_TAUWFBAR + EA_TAUWF[-1] * EA_RHOTAUWF + EA_EPSTAUWF[x]

	EA_CY[0] = EA_C[0] / (EA_Y[0] * EA_PY[0])

	EA_IY[0] = EA_PI[0] * EA_I[0] / (EA_Y[0] * EA_PY[0])

	EA_IMY[0] = EA_PIM[0] * EA_IM[0] / (EA_Y[0] * EA_PY[0])

	EA_IMCY[0] = EA_IMC[0] * EA_PIM[0] / (EA_Y[0] * EA_PY[0])

	EA_IMIY[0] = EA_PIM[0] * EA_IMI[0] / (EA_Y[0] * EA_PY[0])

	EA_BY[0] = EA_B[0] / (EA_PYBAR * EA_YBAR)

	EA_TY[0] = EA_T[0] / (EA_PYBAR * EA_YBAR)

	EA_YGAP[0] = EA_Y[0] / EA_YBAR - 1

	EA_YGROWTH[0] = EA_Y[0] / EA_Y[-1]

	EA_YSHARE[0] = EA_Y[0] * EA_PY[0] * EA_SIZE / EA_RER[0] / (EA_Y[0] * EA_PY[0] * EA_SIZE / EA_RER[0] + US_Y[0] * US_SIZE * US_PY[0] / US_RER)

	EA_EPSILONM[0] = ( - 0.125) / (EA_R[0] * (EA_R[0] + EA_R[0] * EA_GAMMAV2 - 1))

	US_UTILI[0] = 1 / (1 - US_SIGMA) * (US_CI[0] - US_KAPPA * US_CI[-1]) ^ (1 - US_SIGMA) - 1 / (1 + US_ZETA) * US_NI[0] ^ (1 + US_ZETA) + US_BETA * US_UTILI[1]

	US_LAMBDAI[0] * (1 + US_TAUC[0] + US_GAMMAVI[0] + US_VI[0] * US_GAMMAVIDER[0]) = (US_CI[0] - US_KAPPA * US_CI[-1]) ^ (-US_SIGMA)

	US_R[0] = US_LAMBDAI[0] * US_BETA ^ (-1) / US_LAMBDAI[1] * US_PIC[1]

	US_GAMMAVIDER[0] * US_VI[0] ^ 2 = 1 - US_BETA * US_LAMBDAI[1] / (US_LAMBDAI[0] * US_PIC[1])

	US_VI[0] = US_CI[0] * (1 + US_TAUC[0]) / US_MI[0]

	US_GAMMAVI[0] = US_VI[0] * US_GAMMAV1 + US_GAMMAV2 / US_VI[0] - 2 * (US_GAMMAV1 * US_GAMMAV2) ^ 0.5

	US_GAMMAVIDER[0] = US_GAMMAV1 - US_GAMMAV2 * US_VI[0] ^ (-2)

	US_KI[0] = (1 - US_DELTA) * US_KI[-1] + (1 - US_GAMMAI[-1]) * US_II[-1]

	US_GAMMAI[0] = US_GAMMAI1 / 2 * (US_II[0] / US_II[-1] - 1) ^ 2

	US_GAMMAIDER[0] = US_GAMMAI1 * (US_II[0] / US_II[-1] - 1) / US_II[-1]

	US_GAMMAU[0] = ((US_DELTA + US_BETA ^ (-1) - 1) * US_QBAR - US_DELTA * US_TAUKBAR * US_PIBAR) / (US_PIBAR * (1 - US_TAUKBAR)) * (US_U[0] - 1) + US_GAMMAU2 / 2 * (US_U[0] - 1) ^ 2

	US_GAMMAUDER[0] = ((US_DELTA + US_BETA ^ (-1) - 1) * US_QBAR - US_DELTA * US_TAUKBAR * US_PIBAR) / (US_PIBAR * (1 - US_TAUKBAR)) + (US_U[0] - 1) * US_GAMMAU2

	US_RK[0] = US_GAMMAUDER[0] * US_PI[0]

	US_PI[0] = US_Q[0] * (1 - US_GAMMAI[0] - US_II[0] * US_GAMMAIDER[0]) + US_BETA * US_LAMBDAI[1] / US_LAMBDAI[0] * US_Q[1] * US_GAMMAIDER[1] * US_II[1] ^ 2 / US_II[0]

	US_Q[0] = US_BETA * US_LAMBDAI[1] / US_LAMBDAI[0] * ((1 - US_TAUK[1]) * (US_RK[1] * US_U[1] - US_GAMMAU[1] * US_PI[1]) + US_PI[1] * US_DELTA * US_TAUK[1] + (1 - US_DELTA) * US_Q[1])

	US_WITILDE[0] ^ (1 + US_ZETA * US_ETAI) = US_ETAI / (US_ETAI - 1) * US_FI[0] / US_GI[0]

	US_FI[0] = US_WI[0] ^ ((1 + US_ZETA) * US_ETAI) * US_NDI[0] ^ (1 + US_ZETA) + US_BETA * US_XII * (US_PIC[1] / (US_PIC[0] ^ US_CHII * US_PI4TARGET ^ (0.25 * (1 - US_CHII)))) ^ ((1 + US_ZETA) * US_ETAI) * US_FI[1]

	US_GI[0] = US_NDI[0] * US_LAMBDAI[0] * (1 - US_TAUN[0] - US_TAUWH[0]) * US_WI[0] ^ US_ETAI + US_BETA * US_XII * (US_PIC[1] / (US_PIC[0] ^ US_CHII * US_PI4TARGET ^ (0.25 * (1 - US_CHII)))) ^ (US_ETAI - 1) * US_GI[1]

	US_WI[0] ^ (1 - US_ETAI) = (1 - US_XII) * US_WITILDE[0] ^ (1 - US_ETAI) + US_XII * US_WI[-1] ^ (1 - US_ETAI) * (US_PI4TARGET ^ (0.25 * (1 - US_CHII)) * US_PIC[-1] ^ US_CHII / US_PIC[0]) ^ (1 - US_ETAI)

	US_UTILJ[0] = 1 / (1 - US_SIGMA) * (US_CJ[0] - US_KAPPA * US_CJ[-1]) ^ (1 - US_SIGMA) - 1 / (1 + US_ZETA) * US_NJ[0] ^ (1 + US_ZETA) + US_BETA * US_UTILJ[1]

	US_CJ[0] * (1 + US_TAUC[0] + US_GAMMAVJ[0]) + US_MJ[0] = US_NJ[0] * (1 - US_TAUN[0] - US_TAUWH[0]) * US_WJ[0] + US_TRJ[0] - US_TJ[0] + US_MJ[-1] * US_PIC[0] ^ (-1)

	US_LAMBDAJ[0] * (1 + US_TAUC[0] + US_GAMMAVJ[0] + US_VJ[0] * US_GAMMAVJDER[0]) = (US_CJ[0] - US_KAPPA * US_CJ[-1]) ^ (-US_SIGMA)

	US_GAMMAVJDER[0] * US_VJ[0] ^ 2 = 1 - US_BETA * US_LAMBDAJ[1] / (US_PIC[1] * US_LAMBDAJ[0])

	US_VJ[0] = (1 + US_TAUC[0]) * US_CJ[0] / US_MJ[0]

	US_GAMMAVJ[0] = US_GAMMAV1 * US_VJ[0] + US_GAMMAV2 / US_VJ[0] - 2 * (US_GAMMAV1 * US_GAMMAV2) ^ 0.5

	US_GAMMAVJDER[0] = US_GAMMAV1 - US_GAMMAV2 * US_VJ[0] ^ (-2)

	US_WJTILDE[0] ^ (1 + US_ZETA * US_ETAJ) = US_ETAJ / (US_ETAJ - 1) * US_FJ[0] / US_GJ[0]

	US_FJ[0] = US_WJ[0] ^ ((1 + US_ZETA) * US_ETAJ) * US_NDJ[0] ^ (1 + US_ZETA) + US_BETA * US_XIJ * (US_PIC[1] / (US_PIC[0] ^ US_CHIJ * US_PI4TARGET ^ (0.25 * (1 - US_CHIJ)))) ^ ((1 + US_ZETA) * US_ETAJ) * US_FJ[1]

	US_GJ[0] = US_NDJ[0] * (1 - US_TAUN[0] - US_TAUWH[0]) * US_LAMBDAJ[0] * US_WJ[0] ^ US_ETAJ + US_BETA * US_XIJ * (US_PIC[1] / (US_PIC[0] ^ US_CHIJ * US_PI4TARGET ^ (0.25 * (1 - US_CHIJ)))) ^ (US_ETAJ - 1) * US_GJ[1]

	US_WJ[0] ^ (1 - US_ETAJ) = (1 - US_XIJ) * US_WJTILDE[0] ^ (1 - US_ETAJ) + US_XIJ * US_WJ[-1] ^ (1 - US_ETAJ) * (US_PI4TARGET ^ (0.25 * (1 - US_CHIJ)) * US_PIC[-1] ^ US_CHIJ / US_PIC[0]) ^ (1 - US_ETAJ)

	US_YS[0] = US_Z[0] * US_KD[0] ^ US_ALPHA * US_ND[0] ^ (1 - US_ALPHA) - US_PSIBAR

	US_RK[0] = US_ALPHA * (US_YS[0] + US_PSIBAR) / US_KD[0] * US_MC[0]

	US_MC[0] = 1 / (US_Z[0] * US_ALPHA ^ US_ALPHA * (1 - US_ALPHA) ^ (1 - US_ALPHA)) * US_RK[0] ^ US_ALPHA * ((1 + US_TAUWF[0]) * US_W[0]) ^ (1 - US_ALPHA)

	US_NDI[0] = US_ND[0] * (1 - US_OMEGA) * (US_WI[0] / US_W[0]) ^ (-US_ETA)

	US_NDJ[0] = US_ND[0] * US_OMEGA * (US_WJ[0] / US_W[0]) ^ (-US_ETA)

	US_ND[0] ^ (1 - 1 / US_ETA) = (1 - US_OMEGA) ^ (1 / US_ETA) * US_NDI[0] ^ (1 - 1 / US_ETA) + US_OMEGA ^ (1 / US_ETA) * US_NDJ[0] ^ (1 - 1 / US_ETA)

	US_D[0] = US_Y[0] * US_PY[0] - US_RK[0] * US_KD[0] - US_ND[0] * (1 + US_TAUWF[0]) * US_W[0]

	US_PHTILDE[0] / US_PH[0] = US_THETA / (US_THETA - 1) * US_FH[0] / US_GH[0]

	US_FH[0] = US_MC[0] * US_H[0] + US_LAMBDAI[1] * US_BETA * US_XIH / US_LAMBDAI[0] * (US_PIH[1] / (US_PIH[0] ^ US_CHIH * US_PI4TARGET ^ (0.25 * (1 - US_CHIH)))) ^ US_THETA * US_FH[1]

	US_GH[0] = US_PH[0] * US_H[0] + US_LAMBDAI[1] * US_BETA * US_XIH / US_LAMBDAI[0] * (US_PIH[1] / (US_PIH[0] ^ US_CHIH * US_PI4TARGET ^ (0.25 * (1 - US_CHIH)))) ^ (US_THETA - 1) * US_GH[1]

	US_PH[0] ^ (1 - US_THETA) = (1 - US_XIH) * US_PHTILDE[0] ^ (1 - US_THETA) + US_XIH * (US_PH[-1] / US_PIC[0]) ^ (1 - US_THETA) * (US_PI4TARGET ^ (0.25 * (1 - US_CHIH)) * US_PIH[-1] ^ US_CHIH) ^ (1 - US_THETA)

	US_PIH[0] = US_PIC[0] * US_PH[0] / US_PH[-1]

	EA_PIMTILDE[0] / EA_PIM[0] = US_THETA / (US_THETA - 1) * US_FX[0] / US_GX[0]

	US_FX[0] = US_MC[0] * EA_IM[0] * EA_SIZE / US_SIZE + US_LAMBDAI[1] * US_BETA * US_XIX / US_LAMBDAI[0] * (EA_PIIM[1] / (EA_PIIM[0] ^ US_CHIX * US_PI4TARGET ^ (0.25 * (1 - US_CHIX)))) ^ US_THETA * US_FX[1]

	US_GX[0] = EA_IM[0] * EA_SIZE * EA_PIM[0] * USEA_RER[0] / US_SIZE + US_LAMBDAI[1] * US_BETA * US_XIX / US_LAMBDAI[0] * (EA_PIIM[1] / (EA_PIIM[0] ^ US_CHIX * US_PI4TARGET ^ (0.25 * (1 - US_CHIX)))) ^ (US_THETA - 1) * US_GX[1]

	EA_PIM[0] ^ (1 - US_THETA) = (1 - US_XIX) * EA_PIMTILDE[0] ^ (1 - US_THETA) + US_XIX * (EA_PIM[-1] / EA_PIC[0]) ^ (1 - US_THETA) * (EA_PIIM[-1] ^ US_CHIX * EA_PI4TARGET ^ (0.25 * (1 - US_CHIH))) ^ (1 - US_THETA)

	EA_PIIM[0] = EA_PIC[0] * EA_PIM[0] / EA_PIM[-1]

	USEA_RER[0] = US_RER / EA_RER[0]

	US_QC[0] ^ ((US_MUC - 1) / US_MUC) = US_NUC ^ (1 / US_MUC) * US_HC[0] ^ (1 - 1 / US_MUC) + (1 - US_NUC) ^ (1 / US_MUC) * ((1 - US_GAMMAIMC[0]) * US_IMC[0]) ^ (1 - 1 / US_MUC)

	1 = US_NUC * US_PH[0] ^ (1 - US_MUC) + (1 - US_NUC) * (US_PIM[0] / US_GAMMAIMCDAG[0]) ^ (1 - US_MUC)

	US_HC[0] = US_QC[0] * US_NUC * US_PH[0] ^ (-US_MUC)

	US_GAMMAIMC[0] = US_GAMMAIMC1 / 2 * (US_IMC[0] / US_QC[0] / (US_IMC[-1] / US_QC[-1]) - 1) ^ 2

	US_GAMMAIMCDAG[0] = 1 - US_GAMMAIMC[0] - US_IMC[0] * US_GAMMAIMC1 * (US_IMC[0] / US_QC[0] / (US_IMC[-1] / US_QC[-1]) - 1) / US_QC[0] / (US_IMC[-1] / US_QC[-1])

	US_QI[0] ^ ((US_MUI - 1) / US_MUI) = US_NUI ^ (1 / US_MUI) * US_HI[0] ^ (1 - 1 / US_MUI) + (1 - US_NUI) ^ (1 / US_MUI) * ((1 - US_GAMMAIMI[0]) * US_IMI[0]) ^ (1 - 1 / US_MUI)

	US_PI[0] ^ (1 - US_MUI) = US_NUI * US_PH[0] ^ (1 - US_MUI) + (1 - US_NUI) * (US_PIM[0] / US_GAMMAIMIDAG[0]) ^ (1 - US_MUI)

	US_HI[0] = US_QI[0] * US_NUI * (US_PH[0] / US_PI[0]) ^ (-US_MUI)

	US_GAMMAIMI[0] = US_GAMMAIMI1 / 2 * (US_IMI[0] / US_QI[0] / (US_IMI[-1] / US_QI[-1]) - 1) ^ 2

	US_GAMMAIMIDAG[0] = 1 - US_GAMMAIMI[0] - US_IMI[0] * US_GAMMAIMI1 * (US_IMI[0] / US_QI[0] / (US_IMI[-1] / US_QI[0]) - 1) / US_QI[0] / (US_IMI[-1] / US_QI[-1])

	US_PH[-1] * US_G[-1] + US_TR[-1] + US_B[-1] * US_PIC[-1] ^ (-1) + US_PIC[-1] ^ (-1) * US_M[-2] = US_TAUC[-1] * US_C[-1] + (US_TAUN[-1] + US_TAUWH[-1]) * (US_WI[-1] * US_NDI[-1] + US_WJ[-1] * US_NDJ[-1]) + US_TAUWF[-1] * US_W[-1] * US_ND[-1] + US_TAUK[-1] * (US_RK[-1] * US_U[-1] - (US_DELTA + US_GAMMAU[-1]) * US_PI[-1]) * US_K[-1] + US_TAUD[-1] * US_D[-1] + US_T[-1] + US_R[-1] ^ (-1) * US_B[0] + US_M[-1]

	US_PH[0] * US_G[0] = US_GY[0] * US_PYBAR * US_YBAR

	US_TR[0] = US_YBAR * US_PYBAR * US_TRY[0]

	US_T[0] / (US_PYBAR * US_YBAR) = US_PHITB * (US_B[0] / (US_PYBAR * US_YBAR) - US_BYTARGET)

	US_TI[0] = US_T[0] * US_UPSILONT

	US_TRI[0] = US_TR[0] * US_UPSILONTR

	US_PIC4[0] = US_PIC[0] * US_PIC[-1] * US_PIC[-2] * US_PIC[-3]

	US_RR[0] - 1 = US_R[0] / US_PIC[1] - 1

	US_C[0] = US_CI[0] * (1 - US_OMEGA) + US_CJ[0] * US_OMEGA

	US_M[0] = US_MI[0] * (1 - US_OMEGA) + US_MJ[0] * US_OMEGA

	US_K[0] = US_KI[0] * (1 - US_OMEGA)

	US_I[0] = US_II[0] * (1 - US_OMEGA)

	US_TRJ[0] = US_TR[0] * 1 / US_OMEGA - US_TRI[0] * (1 - US_OMEGA) / US_OMEGA

	US_TJ[0] = US_T[0] * 1 / US_OMEGA - US_TI[0] * (1 - US_OMEGA) / US_OMEGA

	US_GAMMAV[0] = US_GAMMAVI[0] * US_CI[0] * (1 - US_OMEGA) + US_GAMMAVJ[0] * US_CJ[0] * US_OMEGA

	US_NI[0] = US_NDI[0] * US_SI[0]

	US_SI[0] = (1 - US_XII) * (US_WITILDE[0] / US_WI[0]) ^ (-US_ETAI) + US_XII * (US_WI[-1] / US_WI[0]) ^ (-US_ETAI) * (US_PIC[0] / (US_PI4TARGET ^ (0.25 * (1 - US_CHII)) * US_PIC[-1] ^ US_CHII)) ^ US_ETAI * US_SI[-1]

	US_NJ[0] = US_NDJ[0] * US_SJ[0]

	US_SJ[0] = (1 - US_XIJ) * (US_WJTILDE[0] / US_WJ[0]) ^ (-US_ETAJ) + US_XIJ * (US_WJ[-1] / US_WJ[0]) ^ (-US_ETAJ) * (US_PIC[0] / (US_PI4TARGET ^ (0.25 * (1 - US_CHIJ)) * US_PIC[-1] ^ US_CHIJ)) ^ US_ETAJ * US_SJ[-1]

	US_U[0] * US_K[0] = US_KD[0]

	US_YS[0] = US_H[0] * US_SH[0] + EA_IM[0] * EA_SIZE * US_SX[0] / US_SIZE

	US_H[0] = US_G[0] + US_HC[0] + US_HI[0]

	US_IM[0] = US_IMC[0] + US_IMI[0]

	US_SH[0] = (1 - US_XIH) * (US_PHTILDE[0] / US_PH[0]) ^ (-US_THETA) + US_XIH * (US_PIH[0] / (US_PI4TARGET ^ (0.25 * (1 - US_CHIH)) * US_PIH[-1] ^ US_CHIH)) ^ US_THETA * US_SH[-1]

	US_SX[0] = (1 - US_XIX) * (EA_PIMTILDE[0] / EA_PIM[0]) ^ (-US_THETA) + US_XIX * (EA_PIIM[0] / (US_PI4TARGET ^ (0.25 * (1 - US_CHIH)) * EA_PIIM[-1] ^ US_CHIX)) ^ US_THETA * US_SX[-1]

	US_QC[0] = US_C[0] + US_GAMMAV[0]

	US_QI[0] = US_I[0] + US_GAMMAU[0] * US_K[0]

	US_Y[0] * US_PY[0] = EA_IM[0] * EA_SIZE * EA_PIM[0] * USEA_RER[0] / US_SIZE + US_PH[0] * US_G[0] + US_QC[0] + US_PI[0] * US_QI[0] - US_PIM[0] * ((1 - US_GAMMAIMC[0]) * US_IMC[0] / US_GAMMAIMCDAG[0] + (1 - US_GAMMAIMI[0]) * US_IMI[0] / US_GAMMAIMIDAG[0])

	US_Y[0] = US_YS[0]

	log(US_Z[0]) = (1 - US_RHOZ) * log(US_ZBAR) + US_RHOZ * log(US_Z[-1]) + US_EPSZ[x]

	US_GY[0] = (1 - US_RHOG) * US_GYBAR + US_RHOG * US_GY[-1] + US_EPSG[x]

	US_TRY[0] = (1 - US_RHOTR) * US_TRYBAR + US_RHOTR * US_TRY[-1] + US_EPSTR[x]

	US_TAUC[0] = (1 - US_RHOTAUC) * US_TAUCBAR + US_TAUC[-1] * US_RHOTAUC + US_EPSTAUC[x]

	US_TAUD[0] = (1 - US_RHOTAUD) * US_TAUDBAR + US_TAUD[-1] * US_RHOTAUD + US_EPSTAUD[x]

	US_TAUK[0] = US_TAUKBAR * (1 - US_RHOTAUK) + US_TAUK[-1] * US_RHOTAUK + US_EPSTAUK[x]

	US_TAUN[0] = (1 - US_RHOTAUN) * US_TAUNBAR + US_TAUN[-1] * US_RHOTAUN + US_EPSTAUN[x]

	US_TAUWH[0] = (1 - US_RHOTAUWH) * US_TAUWHBAR + US_TAUWH[-1] * US_RHOTAUWH + US_EPSTAUWH[x]

	US_TAUWF[0] = (1 - US_RHOTAUWF) * US_TAUWFBAR + US_TAUWF[-1] * US_RHOTAUWF + US_EPSTAUWF[x]

	US_CY[0] = US_C[0] / (US_Y[0] * US_PY[0])

	US_IY[0] = US_PI[0] * US_I[0] / (US_Y[0] * US_PY[0])

	US_IMY[0] = US_PIM[0] * US_IM[0] / (US_Y[0] * US_PY[0])

	US_IMCY[0] = US_PIM[0] * US_IMC[0] / (US_Y[0] * US_PY[0])

	US_IMIY[0] = US_PIM[0] * US_IMI[0] / (US_Y[0] * US_PY[0])

	US_BY[0] = US_B[0] / (US_PYBAR * US_YBAR)

	US_TY[0] = US_T[0] / (US_PYBAR * US_YBAR)

	US_YGAP[0] = US_Y[0] / US_YBAR - 1

	US_YGROWTH[0] = US_Y[0] / US_Y[-1]

	US_YSHARE[0] = US_Y[0] * US_SIZE * US_PY[0] / US_RER / (EA_Y[0] * EA_PY[0] * EA_SIZE / EA_RER[0] + US_Y[0] * US_SIZE * US_PY[0] / US_RER)

	US_EPSILONM[0] = ( - 0.125) / (US_R[0] * (US_R[0] + US_R[0] * US_GAMMAV2 - 1))

	1 = EA_LAMBDAI[1] * EA_BETA * US_R[0] * (1 - EA_GAMMAB[0]) / EA_LAMBDAI[0] * EA_RERDEP[1] / US_PIC[1]

	EA_GAMMAB[0] = EA_GAMMAB1 * (exp(EA_RER[0] * EA_BF[0] / US_PIC[0] / (EA_Y[0] * EA_PY[0]) - EA_BFYTARGET) - 1) - EA_RP[0]

	EA_RP[0] = EA_RHORP * EA_RP[-1] + EA_EPSRP[x]

	EA_RERDEP[0] = EA_RER[0] / EA_RER[-1]

	EA_TOT[0] = EA_PIM[0] / (US_PIM[0] * EA_RER[0])

	EA_TB[0] = US_IM[0] * US_SIZE * US_PIM[0] * EA_RER[0] / EA_SIZE - EA_PIM[0] * EA_IM[0]

	EA_BF[0] / US_R[-1] = EA_BF[-1] + EA_TB[-1] / EA_RER[-1]

	EA_SIZE * EA_BF[0] + US_SIZE * US_BF[0] = 0

end


@parameters NAWM_EAUS_2008 verbose = true begin
	EA_RRSTAR = 1 / EA_BETA

	US_RRSTAR = 1 / US_BETA

	EA_SIZE = 0.4194

	EA_OMEGA = 0.25

	EA_BETA = 0.992638

	EA_SIGMA = 2.00

	EA_KAPPA = 0.60

	EA_ZETA = 2.00

	EA_DELTA = 0.025

	EA_ETA = 6.00

	EA_ETAI = 6.00

	EA_ETAJ = 6.00

	EA_XII = 0.75

	EA_XIJ = 0.75

	EA_CHII = 0.75

	EA_CHIJ = 0.75

	EA_ALPHA = 0.30

	EA_PSI = 0.20

	EA_THETA = 6.00

	EA_XIH = 0.90

	EA_XIX = 0.30

	EA_CHIH = 0.50

	EA_CHIX = 0.50

	EA_NUC = 0.919622

	EA_MUC = 1.5

	EA_NUI = 0.418629

	EA_MUI = 1.5

	EA_GAMMAV1 = 0.289073

	EA_GAMMAV2 = 0.150339

	EA_GAMMAI1 = 3.00

	EA_GAMMAU1     = 0.032765

	EA_GAMMAU2 = 0.007

	EA_GAMMAIMC1 = 2.50

	EA_GAMMAIMI1 = 0.00

	EA_GAMMAB1 = 0.01

	EA_BYTARGET = 2.40

	EA_PHITB = 0.10

	EA_GYBAR = 0.18

	EA_TRYBAR = 0.195161

	EA_TAUCBAR = 0.183

	EA_TAUKBAR = 0.184123

	EA_TAUNBAR = 0.122

	EA_TAUWHBAR = 0.118

	EA_TAUWFBAR = 0.219

	EA_UPSILONT = 1.20

	EA_UPSILONTR = 0.6666666666666666

	EA_PI4TARGET = 1.02

	EA_PHIRR = 0.95

	EA_PHIRPI = 2.00

	EA_PHIRGY = 0.10

	EA_interest_EXOG=EA_BETA^(-1)*EA_PI4TARGET^(1/4)

	EA_BFYTARGET = 0.00

	EA_RHOZ = 0.90

	EA_RHOR        = 0.90

	EA_RHOG = 0.90

	EA_RHOTR = 0.90

	EA_RHOTAUC = 0.90

	EA_RHOTAUK = 0.90

	EA_RHOTAUN = 0.90

	EA_RHOTAUD = 0.90

	EA_RHOTAUWH = 0.90

	EA_RHOTAUWF = 0.90

	EA_PYBAR = 1.00645740523434

	EA_YBAR = 3.62698111871356

	EA_RHORP = 0.9

	EA_PIBAR = 0.961117319822928

	EA_PSIBAR = 0.725396223742712

	EA_QBAR = 0.961117319822928

	EA_TAUDBAR = 0

	EA_ZBAR = 1

	US_SIZE = 0.5806

	US_OMEGA = 0.25

	US_BETA = 0.992638

	US_SIGMA = 2.00

	US_KAPPA = 0.60

	US_ZETA = 2.00

	US_DELTA = 0.025

	US_ETA = 6.00

	US_ETAI = 6.00

	US_ETAJ = 6.00

	US_XII = 0.75

	US_XIJ = 0.75

	US_CHII = 0.75

	US_CHIJ = 0.75

	US_ALPHA = 0.30

	US_PSI = 0.20

	US_THETA = 6.00

	US_XIH = 0.90

	US_XIX = 0.30

	US_CHIH = 0.50

	US_CHIX = 0.50

	US_NUC = 0.899734

	US_MUC = 1.5

	US_NUI = 0.673228

	US_MUI = 1.5

	US_GAMMAV1 = 0.028706

	US_GAMMAV2 = 0.150339

	US_GAMMAI1 = 3.00

	US_GAMMAU1     = 0.034697

	US_GAMMAU2 = 0.007

	US_GAMMAIMC1 = 2.50

	US_GAMMAIMI1 = 0.00

	US_GAMMAB1     = 0.01

	US_BYTARGET = 2.40

	US_PHITB = 0.10

	US_GYBAR = 0.16

	US_TRYBAR = 0.079732

	US_TAUCBAR = 0.077

	US_TAUKBAR = 0.184123

	US_TAUNBAR = 0.154

	US_TAUWHBAR = 0.071

	US_TAUWFBAR = 0.071

	US_UPSILONT = 1.20

	US_UPSILONTR = 0.6666666666666666

	US_PI4TARGET = 1.02

	US_PHIRR = 0.95

	US_PHIRPI = 2.00

	US_PHIRGY = 0.10

	US_interest_EXOG=US_BETA^(-1)*US_PI4TARGET^(1/4)

	US_RHOZ = 0.90

	US_RHOR        = 0.90

	US_RHOG = 0.90

	US_RHOTR = 0.90

	US_RHOTAUC = 0.90

	US_RHOTAUK = 0.90

	US_RHOTAUN = 0.90

	US_RHOTAUD = 0.90

	US_RHOTAUWH = 0.90

	US_RHOTAUWF = 0.90

	US_PYBAR = 0.992282866960427

	US_TAUDBAR = 0

	US_YBAR = 3.92445610588497

	US_PIBAR = 1.01776829477927

	US_PSIBAR = 0.784891221176995

	US_QBAR = 1.01776829477927

	US_ZBAR = 1

	US_RER = 1

	# Help steady state solver
	# EA_FI > 100

	# US_FI > 100

	# EA_FH	> 20
	
	# EA_GH	> 20
	
	# EA_K	> 20
	
	# EA_KD	> 20
	
	# EA_KI	> 20
	
	# US_FH	> 20
	
	# US_FJ	> 20
	
	# US_GH	> 20
	
	# US_K	> 20
	
	# US_KD	> 20
	
	# US_KI	> 20
	
	# US_GI > 10

	# US_GJ > 10
end


# Block: 7 - Solved using lm_ar, iterations: 112, transformer level: 2 and previous best non-converged solution; maximum residual = 9.38953803597542e-10

#lm_kyf is slow on NAWM
# Block: 1 - Solved using lm_kyf, iterations: 4, transformer level: 2 and previous best non-converged solution; maximum residual = 2.0763168961934753e-11
# Block: 2 - Solved using lm_kyf, iterations: 4, transformer level: 2 and previous best non-converged solution; maximum residual = 2.0763168961934753e-11
# Block: 3 - Solved using lm_kyf, iterations: 7, transformer level: 2 and previous best non-converged solution; maximum residual = 6.73594513500575e-12
# Block: 4 - Solved using lm_kyf, iterations: 7, transformer level: 2 and previous best non-converged solution; maximum residual = 6.73594513500575e-12
# Block: 5 - Solved using lm_kyf, iterations: 5, transformer level: 2 and previous best non-converged solution; maximum residual = 5.387754020702573e-9
# Block: 6 - Solved using lm_kyf, iterations: 5, transformer level: 2 and previous best non-converged solution; maximum residual = 5.387754020702573e-9
# Block: 7 - Solved using lm_kyf, iterations: 68092, transformer level: 2 and starting point: 1.5; maximum residual = 1.1652900866465643e-12


# tr doesnt work NAWM
# Block: 1 - Solved using tr, iterations: 4, transformer level: 2 and previous best non-converged solution; maximum residual = 2.3722801500980495e-11
# Block: 2 - Solved using tr, iterations: 4, transformer level: 2 and previous best non-converged solution; maximum residual = 2.3722801500980495e-11
# Block: 3 - Solved using tr, iterations: 5, transformer level: 2 and previous best non-converged solution; maximum residual = 5.249277235108707e-9
# Block: 4 - Solved using tr, iterations: 5, transformer level: 2 and previous best non-converged solution; maximum residual = 5.249277235108707e-9
# Block: 5 - Solved using tr, iterations: 14, transformer level: 2 and previous best non-converged solution; maximum residual = 4.225908512012211e-11
# Block: 6 - Solved using tr, iterations: 14, transformer level: 2 and previous best non-converged solution; maximum residual = 4.225908512012211e-11


# dogleg also rather slow
# Block: 1 - Solved using dogleg, iterations: 4, transformer level: 3 and starting point: 1.1; maximum residual = 1.311661890213145e-11
# Block: 2 - Solved using dogleg, iterations: 12, transformer level: 3 and starting point: 0.75; maximum residual = 1.885918301299666e-9
# Block: 3 - Solved using dogleg, iterations: 9, transformer level: 3 and previous best non-converged solution; maximum residual = 3.1086244689504383e-15
# Block: 4 - Solved using dogleg, iterations: 4, transformer level: 3 and starting point: 1.1; maximum residual = 1.311661890213145e-11
# Block: 5 - Solved using dogleg, iterations: 11, transformer level: 3 and previous best non-converged solution; maximum residual = 3.5128150388530344e-14
# Block: 6 - Solved using dogleg, iterations: 14, transformer level: 3 and starting point: -0.5; maximum residual = 2.0028423364237824e-12

# nr slow as well
# Block: 1 - Solved using nr, iterations: 4, transformer level: 3 and previous best non-converged solution; maximum residual = 7.199996154838573e-11
# Block: 2 - Solved using nr, iterations: 9, transformer level: 3 and previous best non-converged solution; maximum residual = 2.220446049250313e-15
# Block: 3 - Solved using nr, iterations: 9, transformer level: 3 and previous best non-converged solution; maximum residual = 2.220446049250313e-15
# Block: 4 - Solved using nr, iterations: 4, transformer level: 3 and previous best non-converged solution; maximum residual = 7.199996154838573e-11
# Block: 5 - Solved using nr, iterations: 6, transformer level: 3 and previous best non-converged solution; maximum residual = 3.1086244689504383e-15
# Block: 6 - Solved using nr, iterations: 6, transformer level: 3 and previous best non-converged solution; maximum residual = 3.1086244689504383e-15

# lm_ar 
# Block: 1 - Solved using lm_ar, iterations: 3, transformer level: 4 and previous best non-converged solution; maximum residual = 4.440892098500626e-16
# Block: 2 - Solved using lm_ar, iterations: 3, transformer level: 4 and previous best non-converged solution; maximum residual = 4.440892098500626e-16
# Block: 3 - Solved using lm_ar, iterations: 5, transformer level: 4 and previous best non-converged solution; maximum residual = 2.4158453015843406e-13
# Block: 4 - Solved using lm_ar, iterations: 5, transformer level: 4 and previous best non-converged solution; maximum residual = 2.4158453015843406e-13
# Block: 5 - Solved using lm_ar, iterations: 4, transformer level: 4 and previous best non-converged solution; maximum residual = 3.400058012914542e-16
# Block: 6 - Solved using lm_ar, iterations: 4, transformer level: 4 and previous best non-converged solution; maximum residual = 3.400058012914542e-16
# Block: 7 - Solved using lm_ar, iterations: 338, transformer level: 4 and previous best non-converged solution; maximum residual = 9.278267043555388e-11

# lm_ar no 10 lower bounds 
# Block: 1 - Solved using lm_ar, iterations: 3, transformer level: 4 and previous best non-converged solution; maximum residual = 4.440892098500626e-16
# Block: 2 - Solved using lm_ar, iterations: 3, transformer level: 4 and previous best non-converged solution; maximum residual = 4.440892098500626e-16
# Block: 3 - Solved using lm_ar, iterations: 5, transformer level: 4 and previous best non-converged solution; maximum residual = 2.4158453015843406e-13
# Block: 4 - Solved using lm_ar, iterations: 5, transformer level: 4 and previous best non-converged solution; maximum residual = 2.4158453015843406e-13
# Block: 5 - Solved using lm_ar, iterations: 4, transformer level: 4 and previous best non-converged solution; maximum residual = 3.400058012914542e-16
# Block: 6 - Solved using lm_ar, iterations: 4, transformer level: 4 and previous best non-converged solution; maximum residual = 3.400058012914542e-16
# Block: 7 - Solved using lm_ar, iterations: 778, transformer level: 4 and previous best non-converged solution; maximum residual = 8.15703060652595e-12


# lm_ar no 20 lower bounds 