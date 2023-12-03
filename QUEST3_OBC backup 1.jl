### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ d93e2dfc-a0be-4dba-8526-b6bd0301a49a
import Pkg; Pkg.add("MacroModelling")

# ╔═╡ 3e1ebba0-9229-11ee-230d-251f23eaf3f4
using MacroModelling

# ╔═╡ 0f0b1b84-7899-4013-8005-a52ac8664714


# ╔═╡ 2de2b8b4-5d8e-4339-bdcc-9c5700dcba2c


# ╔═╡ 26613261-5fd5-46f4-993d-3442f9f0a6d1
@model QUEST3_2009_OBC begin
	interest[0] = ((1 + E_INOM[0]) ^ 4 - interestq_exog ^ 4) / interestq_exog ^ 4

	inflation[0] = 0.25 * (inflationq[0] + inflationq[-1] + inflationq[-2] + inflationq[-3])

	inflationq[0] = (1 + 4 * E_PHIC[0] - inflationannual_exog) / inflationannual_exog

	outputgap[0] = E_LYGAP[0]

	E_INOM[0] = ILAGE * E_INOM[-1] + (1 - ILAGE) * (E_EX_R + GP0 + TINFE * (E_PHIC[0] - GP0) + TYE1 * E_LYGAP[-1]) + TYE2 * (E_LYGAP[0] - E_LYGAP[-1]) + E_ZEPS_M[0]

	exp(E_LUCYN[0]) = exp(E_ZEPS_C[0]) * (exp(E_LCNLCSN[0]) * (1 - HABE / (1 + E_GCNLC[0] - GY0))) ^ (-SIGC) * (1 - OMEGE * exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ KAPPAE) ^ (1 - SIGC)

	exp(E_LUCLCYN[0]) = (1 - OMEGE * exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ KAPPAE) ^ (1 - SIGC) * exp(E_LCLCSN[0]) ^ (-SIGC)

	E_VL[0] = exp(E_ZEPS_L[0]) * OMEGE * KAPPAE * exp(E_ZEPS_C[0]) * (exp(E_LCNLCSN[0]) * (1 - HABE / (1 + E_GCNLC[0] - GY0))) ^ (1 - SIGC) * (1 - OMEGE * exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ KAPPAE) ^ (-SIGC) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ (KAPPAE - 1)

	E_VLLC[0] = exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ (KAPPAE - 1) * OMEGE * KAPPAE * (1 - OMEGE * exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ KAPPAE) ^ (-SIGC) * exp(E_LCLCSN[0]) ^ (1 - SIGC)

	1 / BETAE - 1 = E_INOM[0] + E_GUC[1] - E_PHIC[1]

	E_LUCYN[0] - E_LUCYN[-1] = E_GUC[0] + SIGC * (E_GY[0] - GY0 - E_PHIC[0] + E_PHI[0])

	exp(E_LCLCSN[0]) * (1 + TVAT) = (1 - E_TW[0] - SSC) * E_WS[0] + E_WS[0] * E_TRW[0] #- E_TAXYN[0]

	E_WS[0] = exp(E_LL[0] - E_LYWR[0])

	exp(E_LCSN[0]) = exp(E_LCLCSN[0]) * SLCFLAG * SLC + exp(E_LCNLCSN[0]) * (1 - SLCFLAG * SLC)

	(1 + TVAT) * ((E_VL[0] * (1 - SLC) + E_VLLC[0] * SLC) / (exp(E_LUCYN[0]) * (1 - SLC) + exp(E_LUCLCYN[0]) * SLC)) ^ (1 - WRLAG) * ((1 - E_TW[0] - SSC) / (1 + TVAT) * (THETAE - 1) / THETAE / exp(E_LYWR[-1]) / (1 + E_GY[0] - GY0)) ^ WRLAG = (1 - E_TW[0] - SSC) * (THETAE - 1) / THETAE / exp(E_LYWR[0]) + GAMIFLAG * GAMWE / THETAE / exp(E_LYWR[0]) * (E_WPHI[0] - GP0 - GY0 - (1 - SFWE) * (E_PHI[-1] - GP0)) - GAMWE * BETAE * GAMIFLAG / THETAE / exp(E_LYWR[0]) * (E_WPHI[1] - GP0 - GY0 - (1 - SFWE) * (E_PHI[0] - GP0))

	(1 + E_ZEPS_W[0]) / exp(E_LYWR[0]) = E_ETA[0] * ALPHAE / exp(E_LL[0]) * (1 + E_LOL[0]) - 1 / exp(E_LYWR[0]) * GAMLE * (E_LL[0] - E_LL[-1]) + GAMLE * 1 / exp(E_LYWR[1]) * (1 + E_GY[1] - GY0) / (1 + E_R[0]) * (E_LL[1] - E_LL[0])

	GAMIE * (exp(E_LIK[0]) - (GY0 + DELTAE + GPOP0 + GPCPI0)) + GAMI2E * (E_GI[0] - GY0 - GPCPI0) - GAMI2E / (1 + E_INOM[0]) * (E_GI[1] - GY0 - GPCPI0) = E_Q[0] - 1

	E_ETA[0] * (1 - TP) * (1 - ALPHAE) * exp(E_LYKPPI[0]) = E_Q[0] - (GPCPI0 + 1 - E_R[0] - DELTAE - RPREMK - E_ZEPS_RPREMK[0] - E_PHIPI[1]) * E_Q[1] + (1 - TP) * (A1E * (E_UCAP[0] - UCAP0) + A2E * (E_UCAP[0] - UCAP0) ^ 2)

	exp(E_LYKPPI[0]) * E_ETA[0] * (1 - ALPHAE) = E_UCAP[0] * (A1E + (E_UCAP[0] - UCAP0) * 2 * A2E)

	E_ETA[0] = 1 - (TAUE + E_ZEPS_ETA[0]) - GAMIFLAG * GAMPE * (BETAE * (SFPE * E_PHI[1] + E_PHI[-1] * (1 - SFPE) - GP0) - (E_PHI[0] - GP0))

	E_MRY[0] = (1 + E_INOM[0]) ^ (-ZETE)

	E_GK[0] - (GY0 + GPCPI0) = exp(E_LIK[0]) - (GY0 + DELTAE + GPOP0 + GPCPI0)

	E_GKG[0] - (GY0 + GPCPI0) = exp(E_LIKG[0]) - (GPCPI0 + GY0 + GPOP0 + DELTAGE)

	E_LISN[0] = GPCPI0 + GY0 + E_LIK[0] - E_LYKPPI[0] - E_GK[0]

	E_GY[0] = (1 - ALPHAE) * (E_GK[0] + E_GUCAP[0]) + ALPHAE * (E_GTFP[0] + E_GL[0] * (1 + LOL)) + E_GKG[0] * (1 - ALPHAGE)

	E_R[0] = E_INOM[0] - E_PHI[1]

	E_LYGAP[0] = (1 - ALPHAE) * (log(E_UCAP[0]) - log(E_UCAP0[0])) + ALPHAE * (E_LL[0] - E_LL0[0])

	E_LL0[0] = RHOL0 * E_LL0[-1] + E_LL[0] * (1 - RHOL0)

	E_UCAP0[0] = RHOUCAP0 * E_UCAP0[-1] + E_UCAP[0] * (1 - RHOUCAP0)

	exp(E_LPCP[0]) = (SE + (1 - SE) * exp(E_LPMP[0]) ^ (1 - SIGIME)) ^ (1 / (1 - SIGIME))

	exp(E_LPMP[0]) = (1 + E_ZEPS_ETAM[0] + GAMIFLAG * GAMPME * (GP0 + BETAE * (SFPME * E_PHIM[1] + (1 - SFPME) * E_PHIM[-1] - GP0) - E_PHIM[0])) * exp(E_LER[0]) ^ ALPHAX

	exp(E_LPXP[0]) = 1 + E_ZEPS_ETAX[0] + GAMIFLAG * GAMPXE * (GP0 + BETAE * (SFPXE * E_PHIX[1] + (1 - SFPXE) * E_PHIX[-1] - GP0) - E_PHIX[0])

	1 = exp(E_LCSN[0]) + exp(E_LISN[0]) + exp(E_LIGSN[0]) + exp(E_LGSN[0]) + E_TBYN[0]

	E_TBYN[0] = exp(E_LEXYN[0]) - exp(E_LIMYN[0]) + E_ZEPS_EX[0]

	exp(E_LIMYN[0]) = (exp(E_LCSN[0]) + exp(E_LISN[0]) + exp(E_LIGSN[0]) + exp(E_LGSN[0])) * (1 - SE) * exp(RHOPCPM * (E_LPCP[-1] - E_LPMP[-1]) + (1 - RHOPCPM) * (E_LPCP[0] - E_LPMP[0])) ^ SIGIME * exp(E_LPMP[0] - E_LPCP[0])

	exp(E_LEXYN[0]) = exp(E_LPXP[0]) * (1 - SE) * exp(RHOPWPX * (E_LER[-1] * SE * ALPHAX - E_LPXP[-1]) + (1 - RHOPWPX) * (E_LER[0] * SE * ALPHAX - E_LPXP[0])) ^ SIGEXE * exp(E_LYWY[0]) ^ ALPHAX

	E_INOM[0] = E_INOMW[0] + E_GE[1] - RPREME * E_BWRY[0] + E_ZEPS_RPREME[0]

	E_BWRY[0] = E_TBYN[0] + (1 + E_INOM[0] - E_PHI[1] - E_GY[0] - GPOP0) * E_BWRY[-1]

	exp(E_LBGYN[0]) = exp(E_LIGSN[0]) + exp(E_LGSN[0]) + (1 + E_R[0] - E_GY[0] - GPOP0) * exp(E_LBGYN[-1]) + E_TRW[0] * exp(E_LL[0] - E_LYWR[0]) - E_WS[0] * (E_TW[0] + SSC) - TP * (1 - E_WS[0]) - TVAT * exp(E_LCSN[0]) # 	- E_TAXYN[0]

	E_GG[0] - GY0 = GSLAG * (E_GG[-1] - GY0) + GFLAG * GVECM * (E_LGSN[-1] - log(GSN)) + (E_LYGAP[0] - E_LYGAP[-1]) * GFLAG * G1E + GFLAG * GEXOFLAG * E_ZEPS_G[0] + (1 - GFLAG) * (E_ZEPS_G[0] - E_ZEPS_G[-1])

	E_GIG[0] - GY0 - GPCPI0 = IGSLAG * (E_GIG[-1] - GY0 - GPCPI0) + IGFLAG * IGVECM * (E_LIGSN[-1] - log(IGSN)) + (E_LYGAP[0] - E_LYGAP[-1]) * IGFLAG * IG1E + IGFLAG * IGEXOFLAG * E_ZEPS_IG[0] + (1 - IGFLAG) * (E_ZEPS_IG[0] - E_ZEPS_IG[-1])

	E_GIG[0] - E_GI[0] = E_LIGSN[0] - E_LISN[0] - E_LIGSN[-1] + E_LISN[-1]

	E_TRW[0] = TRSN + TRFLAG * TR1E * (1 - exp(E_LL[0]) - (1 - L0)) + E_ZEPS_TR[0]

	# E_TAXYN[0] - E_TAXYN[-1] = BGADJ1 * (exp(E_LBGYN[-1]) - BGTAR) + BGADJ2 * (exp(E_LBGYN[0]) - exp(E_LBGYN[-1]))

	E_LBGYN[0] = max(BGTAR , - BGADJ1/BGADJ2 * (exp(E_LBGYN[-1]) - BGTAR) + exp(E_LBGYN[-1]))

	E_TW[0] = TW0 * (1 + E_LYGAP[0] * TW1 * TWFLAG)

	E_TRYN[0] = E_TRW[0] * exp(E_LL[0] - E_LYWR[0])

	E_TRTAXYN[0] = E_TRW[0] * exp(E_LL[0] - E_LYWR[0]) #- E_TAXYN[0]

	E_WSW[0] = (1 - E_TW[0] - SSC) * E_WS[0]

	E_INOMW[0] = (1 - RII) * E_EX_INOMW + RII * E_INOMW[-1] + RIP * (E_PHIW[-1] - GPW0) + RIX * (E_GYW[-1] - GYW0) + STD_EPS_INOMW *E_EPS_INOMW[x]

	E_PHIW[0] - GPW0 = RPI * (E_INOMW[-1] - E_EX_INOMW) + (E_PHIW[-1] - GPW0) * RPP + (E_GYW[-1] - GYW0) * RPX + STD_EPS_PW * E_EPS_PW[x]

	E_GYW[0] - GYW0 = (E_INOMW[-1] - E_EX_INOMW) * RXI + (E_PHIW[-1] - GPW0) * RXP + (E_GYW[-1] - GYW0) * RXX + RXY * (E_LYWY[-1] - LYWY0) + STD_EPS_YW * E_EPS_YW[x]

	E_LYWY[0] - E_LYWY[-1] = E_GYW[0] - E_GY[0]

	E_GTFP[0] - GTFP0 = STD_EPS_Y * E_EPS_Y[x]

	E_LOL[0] - LOL = RHOLOL * (E_LOL[-1] - LOL) + STD_EPS_LOL * E_EPS_LOL[x]

	E_PHIPI[0] = GPCPI0 + E_ZEPS_PPI[0]

	E_ZEPS_C[0] = RHOCE * E_ZEPS_C[-1] + STD_EPS_C * E_EPS_C[x]

	E_ZEPS_ETA[0] = RHOETA * E_ZEPS_ETA[-1] + STD_EPS_ETA * E_EPS_ETA[x]

	E_ZEPS_ETAM[0] = RHOETAM * E_ZEPS_ETAM[-1] + STD_EPS_ETAM * E_EPS_ETAM[x]

	E_ZEPS_ETAX[0] = RHOETAX * E_ZEPS_ETAX[-1] + STD_EPS_ETAX * E_EPS_ETAX[x]

	E_ZEPS_EX[0] = RHOEXE * E_ZEPS_EX[-1] + STD_EPS_EX * E_EPS_EX[x]

	E_ZEPS_G[0] = E_ZEPS_G[-1] * RHOGE + STD_EPS_G * E_EPS_G[x]

	E_ZEPS_IG[0] = E_ZEPS_IG[-1] * RHOIG + IGEXOFLAG * STD_EPS_IG * E_EPS_IG[x]

	E_ZEPS_L[0] = RHOLE * E_ZEPS_L[-1] + STD_EPS_L * E_EPS_L[x]

	E_ZEPS_M[0] = STD_EPS_M * E_EPS_M[x]

	E_ZEPS_PPI[0] = STD_EPS_PPI * E_EPS_PPI[x] + RHOPPI1 * E_ZEPS_PPI[-1] + RHOPPI2 * E_ZEPS_PPI[-2] + RHOPPI3 * E_ZEPS_PPI[-3] + RHOPPI4 * E_ZEPS_PPI[-4]

	E_ZEPS_RPREME[0] = RHORPE * E_ZEPS_RPREME[-1] + STD_EPS_RPREME * E_EPS_RPREME[x]

	E_ZEPS_RPREMK[0] = RHORPK * E_ZEPS_RPREMK[-1] + STD_EPS_RPREMK * E_EPS_RPREMK[x]

	E_ZEPS_W[0] = STD_EPS_W * E_EPS_W[x]

	E_ZEPS_TR[0] = RHOTR * E_ZEPS_TR[-1] + TREXOFLAG * STD_EPS_TR * E_EPS_TR[x]

	E_PHIC[0] + E_GC[0] - E_GY[0] - E_PHI[0] = E_LCSN[0] - E_LCSN[-1]

	E_PHIC[0] + E_GCLC[0] - E_GY[0] - E_PHI[0] = E_LCLCSN[0] - E_LCLCSN[-1]

	E_PHIC[0] + E_GCNLC[0] - E_GY[0] - E_PHI[0] = E_LCNLCSN[0] - E_LCNLCSN[-1]

	E_PHIW[0] + E_GE[0] - E_PHI[0] = E_LER[0] - E_LER[-1]

	E_PHIX[0] + E_GEX[0] - E_GY[0] - E_PHI[0] = E_LEXYN[0] - E_LEXYN[-1]

	E_PHIC[0] + E_GG[0] - E_GY[0] - E_PHI[0] = E_LGSN[0] - E_LGSN[-1]

	E_GI[0] - E_GK[-1] = E_LIK[0] - E_LIK[-1]

	E_GIG[0] - E_GKG[-1] = E_LIKG[0] - E_LIKG[-1]

	E_PHIM[0] + E_GIM[0] - E_GY[0] - E_PHI[0] = E_LIMYN[0] - E_LIMYN[-1]

	E_PHIPI[0] + E_GY[0] - E_GK[0] = E_LYKPPI[0] - E_LYKPPI[-1]

	E_GL[0] = E_LL[0] - E_LL[-1]

	E_GTAX[0] - E_GY[0] - E_PHI[0] = 0 #log(E_TAXYN[0]/E_TAXYN[-1])

	E_GTFPUCAP[0] = (1 - ALPHAE) * E_GUCAP[0] + ALPHAE * E_GTFP[0]

	E_GTR[0] - E_GL[0] - E_WRPHI[0] = log(E_TRW[0]/E_TRW[-1])

	E_GUCAP[0] = log(E_UCAP[0]/E_UCAP[-1])

	E_GWRY[0] = E_LYWR[-1] - E_LYWR[0]

	E_GY[0] - E_GYPOT[0] = E_LYGAP[0] - E_LYGAP[-1]

	E_BGYN[0] = exp(E_LBGYN[0])

	E_DBGYN[0] = E_BGYN[0] - E_BGYN[-1]

	E_CLCSN[0] = exp(E_LCLCSN[0])

	E_GSN[0] = exp(E_LGSN[0])

	E_LTRYN[0] = log(E_TRYN[0])

	E_PHIC[0] - E_PHI[0] = E_LPCP[0] - E_LPCP[-1]

	E_PHIM[0] - E_PHI[0] = E_LPMP[0] - E_LPMP[-1]

	E_PHIX[0] - E_PHI[0] = E_LPXP[0] - E_LPXP[-1]

	E_PHI[0] + E_GY[0] - E_WPHI[0] = E_LYWR[0] - E_LYWR[-1]

	E_WPHI[0] = E_PHI[0] + E_WRPHI[0]

	E_GYL[0] = E_GY[0] + GPOP0

	E_GCL[0] = GPOP0 + E_GC[0]

	E_GIL[0] = GPOP0 + E_GI[0]

	E_GGL[0] = GPOP0 + E_GG[0]

	E_GEXL[0] = GPOP0 + E_GEX[0] + DGEX

	E_GIML[0] = GPOP0 + E_GIM[0] + DGIM

	E_PHIML[0] = E_PHIM[0] + DGPM

	E_PHIXL[0] = E_PHIX[0] + DGPX

	E_LCY[0] = E_LCSN[0] - E_LPCP[0]

	E_LGY[0] = E_LGSN[0] - E_LPCP[0]

	E_LWS[0] = E_LL[0] - E_LYWR[0]

end


# ╔═╡ cdf1bbca-26f0-4987-823a-5fa951181143

@parameters QUEST3_2009_OBC begin
	STD_EPS_INOMW = 0.0023

	STD_EPS_PW = 0.0029

	STD_EPS_YW = 0.0044

	STD_EPS_PPI = 0.00312216772065

	STD_EPS_C = 0.0597

	STD_EPS_ETA = 0.1500

	STD_EPS_ETAM = 0.0202

	STD_EPS_ETAX = 0.0648

	STD_EPS_EX = 0.0044

	STD_EPS_G = 0.0048

	STD_EPS_IG = 0.0056

	STD_EPS_L = 0.0283

	STD_EPS_LOL = 0.0048

	STD_EPS_M = 0.0013

	STD_EPS_RPREME = 0.0017

	STD_EPS_RPREMK = 0.0070

	STD_EPS_TR = 0.0022

	STD_EPS_W = 0.0437

	STD_EPS_Y = 0.0121
	
	A2E = 0.0453

	G1E = (-0.0754)

	GAMIE = 76.0366

	GAMI2E = 1.1216

	GAMLE = 58.2083

	GAMPE = 61.4415

	GAMPME = 1.6782

	GAMPXE = 26.1294

	GAMWE = 1.2919

	GSLAG = (-0.4227)

	GVECM = (-0.1567)

	HABE = 0.5634

	HABLE = 0.8089

	IG1E = 0.1497

	IGSLAG = 0.4475

	IGVECM = (-0.1222)

	ILAGE = 0.9009

	KAPPAE = 1.9224

	RHOCE = 0.9144

	RHOETA = 0.1095

	RHOETAM = 0.9557

	RHOETAX = 0.8109

	RHOGE = 0.2983

	RHOIG = 0.8530

	RHOLE = 0.9750

	RHOL0 = 0.9334

	RHOPCPM = 0.6652

	RHOPWPX = 0.2159

	RHORPE = 0.9842

	RHORPK = 0.9148

	RHOUCAP0 = 0.9517

	RPREME = 0.0200

	RPREMK = 0.0245

	SE = 0.8588

	SFPE = 0.8714

	SFPME = 0.7361

	SFPXE = 0.9180

	SFWE = 0.7736

	SIGC = 4.0962

	SIGEXE = 2.5358

	SIGIME = 1.1724

	SLC = 0.3507

	TINFE = 1.9590

	TR1E = 0.9183

	RHOTR = 0.8636

	TYE1 = 0.4274

	TYE2 = 0.0783

	WRLAG = 0.2653

	ALPHAX = 0.5

	BETAE = 0.996

	ALPHAE = 0.52

	ALPHAGE = 0.9

	BGADJ2 = 0.004

	BGADJ1 = 0.001*BGADJ2

	BGTAR = 2.4

	DELTAE = 0.025

	DELTAGE = 0.0125

	DGIM = 0.00738619107021

	DGEX = 0.00738619107021

	DGPM = (-0.00396650612294)

	DGPX = (-0.00396650612294)

	LOL = 0

	RHOEXE = 0.975

	RHOLOL = 0.99

	SSC = 0.2

	TAUE = 0.1000

	THETAE = 1.6

	TP = 0.2

	TRSN = 0.36

	TVAT = 0.2

	TW0 = 0.2

	TW1 = 0.8

	ZETE = 0.4000

	GFLAG = 1

	IGFLAG = 1

	TRFLAG = 1

	TWFLAG = 1

	GEXOFLAG = 1

	IGEXOFLAG = 1

	TREXOFLAG = 1

	SLCFLAG = 1

	GAMIFLAG = 1

	A1E = 0.0669

	OMEGE = 1.4836

	GSN = 0.203

	IGSN = 0.025

	GPCPI0 = 0

	GP0 = 0.005

	GPW0 = GP0

	GPOP0 = 0.00113377677398

	GY0 = 0.003

	GTFP0 = (ALPHAE+ALPHAGE-1)/ALPHAE*GY0-(2-ALPHAE-ALPHAGE)/ALPHAE*GPCPI0

	GYW0 = GY0

	UCAP0 = 1

	L0 = 0.65

	E_EX_R = 1/BETAE-1

	E_EX_INOMW = GP0+E_EX_R

	LYWY0 = 0

	RXY = (-0.0001)

	RII = 0.887131978334279

	RIP = 0.147455589872832

	RIX = 0.120095599681076

	RPI = 0.112067224979767

	RPP = 0.502758194258928

	RPX = 0.082535400836409

	RXI = 0.073730131176521

	RXP = (-0.302655015002645)

	RXX = 0.495000246553550

	RHOPPI1 = 0.24797097628284

	RHOPPI2 = 0.13739098460472

	RHOPPI3 = 0.10483962746747

	RHOPPI4 = 0.09282876044442

	interestq_exog = 1.00901606

	inflationannual_exog = 1.02

end


# ╔═╡ bfc386c1-c979-4e85-baee-ed07c1d7d82e


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
MacroModelling = "687ffad2-3618-405e-ac50-e0f7b9c75e44"

[compat]
MacroModelling = "~0.1.31"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "2e5fd3b9a4da70dc69a734f37b05676dac333ee5"

[[deps.ADTypes]]
git-tree-sha1 = "332e5d7baeff8497b923b730b994fa480601efc7"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.5"

[[deps.AMD]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "45a1272e3f809d36431e57ab22703c6896b8908f"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.5.3"

[[deps.ANSIColoredPrinters]]
git-tree-sha1 = "574baf8110975760d391c710b6341da1afa48d8c"
uuid = "a4c015fc-c6ff-483c-b24f-f7ea428134e9"
version = "0.0.1"

[[deps.AbstractAlgebra]]
deps = ["GroupsCore", "InteractiveUtils", "LinearAlgebra", "MacroTools", "Preferences", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "c3c29bf6363b3ac3e421dc8b2ba8e33bdacbd245"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.32.5"

[[deps.AbstractDifferentiation]]
deps = ["ExprTools", "LinearAlgebra", "Requires"]
git-tree-sha1 = "6a5e61dc899ab116035c18ead4ec890269f3c478"
uuid = "c29ec348-61ec-40c8-8164-b8c60e9d9f3d"
version = "0.6.0"

    [deps.AbstractDifferentiation.extensions]
    AbstractDifferentiationChainRulesCoreExt = "ChainRulesCore"
    AbstractDifferentiationFiniteDifferencesExt = "FiniteDifferences"
    AbstractDifferentiationForwardDiffExt = ["DiffResults", "ForwardDiff"]
    AbstractDifferentiationReverseDiffExt = ["DiffResults", "ReverseDiff"]
    AbstractDifferentiationTrackerExt = "Tracker"
    AbstractDifferentiationZygoteExt = "Zygote"

    [deps.AbstractDifferentiation.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Test"]
git-tree-sha1 = "a7055b939deae2455aa8a67491e034f735dd08d3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.33"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[[deps.AccurateArithmetic]]
deps = ["LinearAlgebra", "Random", "VectorizationBase"]
git-tree-sha1 = "07af26e8d08c211ef85918f3e25d4c0990d20d70"
uuid = "22286c92-06ac-501d-9306-4abd417d9753"
version = "0.3.8"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "02f731463748db57cc2ebfbd9fbc9ce8280d3433"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "247efbccf92448be332d154d6ca56b9fcdd93c31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.6.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisKeys]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "IntervalSets", "InvertedIndices", "LazyStack", "LinearAlgebra", "NamedDims", "OffsetArrays", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "dba0fdaa3a95e591aa9cbe0df9aba41e295a2011"
uuid = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
version = "0.2.13"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.Bijections]]
git-tree-sha1 = "c9b163bd832e023571e86d0b90d9de92a9879088"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.6"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.BlockTriangularForm]]
deps = ["SparseArrays"]
git-tree-sha1 = "64281233ecb50b39fc58d49b880a880203c31a96"
uuid = "adeb47b7-70bf-415a-bb24-c358563e873a"
version = "0.1.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e0af648f0692ec1691b5d094b8724ba1346281cf"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.18.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "c0ae2a86b162fb5d7acc65269b469ff5b8a73594"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonEq]]
git-tree-sha1 = "6b0f0354b8eb954cdba708fb262ef00ee7274468"
uuid = "3709ef60-1bee-4518-9f2f-acd86f176c50"
version = "0.2.1"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "886826d76ea9e72b35fcd000e535588f7b60f21d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "02d2316b7ffceff992f3096ae48c7829a8aa0638"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.3"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.CondaPkg]]
deps = ["JSON3", "Markdown", "MicroMamba", "Pidfile", "Pkg", "Preferences", "TOML"]
git-tree-sha1 = "e81c4263c7ef4eca4d645ef612814d72e9255b41"
uuid = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
version = "0.2.22"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "9a44ddc9e60ee398934b73a5168f5806989e6792"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.11"

[[deps.Coverage]]
deps = ["CoverageTools", "HTTP", "JSON", "LibGit2", "MbedTLS"]
git-tree-sha1 = "4fb5effc927fddc76a213dc4b1871dc41b666686"
uuid = "a2441757-f6aa-5fb2-8edb-039e3f45d037"
version = "1.6.0"

[[deps.CoverageTools]]
git-tree-sha1 = "cc5595feb314d3b226ed765a001a40ca451ad687"
uuid = "c36e975a-824b-4404-a568-ef97ca766997"
version = "1.3.0"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a6c00f894f24460379cb7136633cef54ac9f6f4a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.103"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Documenter]]
deps = ["ANSIColoredPrinters", "AbstractTrees", "Base64", "Dates", "DocStringExtensions", "Downloads", "Git", "IOCapture", "InteractiveUtils", "JSON", "LibGit2", "Logging", "Markdown", "MarkdownAST", "Pkg", "PrecompileTools", "REPL", "RegistryInstances", "SHA", "Test", "Unicode"]
git-tree-sha1 = "43aa88b72dffff46b1b19f66483ea3e2f907c4fa"
uuid = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
version = "1.2.0"

[[deps.DocumenterTools]]
deps = ["AbstractTrees", "Base64", "DocStringExtensions", "Documenter", "FileWatching", "Gumbo", "LibGit2", "OpenSSH_jll", "Sass"]
git-tree-sha1 = "37402c74604d89c94593bc74b3e1ca597494a804"
uuid = "35a29f4d-8980-5a13-9543-d66fff28ecb8"
version = "0.1.18"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "51b4b84d33ec5e0955b55ff4b748b99ce2c3faa9"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.6.7"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "fea68c84ba262b121754539e6ea0546146515d4f"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.5.3"

[[deps.DynarePreprocessor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dc0e7282f378cf98ccad07ae209ccb3d36d235b6"
uuid = "23afba7c-24e5-5ee2-bc2c-b42e07f0492a"
version = "6.3.0+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "28e4e9c4b7b162398ec8004bdabe9a90c78c122d"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.8.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.Git]]
deps = ["Git_jll"]
git-tree-sha1 = "51764e6c2e84c37055e846c516e9015b4a291c7d"
uuid = "d7ba0133-e1db-5d97-8f8c-041e4b3a1eb2"
version = "1.3.0"

[[deps.Git_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "Libiconv_jll", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "d8be4aab0f4e043cc40984e9097417307cce4c03"
uuid = "f8c6e375-362e-5223-8a59-34ff63f689eb"
version = "2.36.1+2"

[[deps.Groebner]]
deps = ["AbstractAlgebra", "Combinatorics", "ExprTools", "Logging", "MultivariatePolynomials", "Primes", "Random", "SIMD", "SnoopPrecompile"]
git-tree-sha1 = "44f595de4f6485ab5ba71fe257b5eadaa3cf161e"
uuid = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
version = "0.4.4"

[[deps.GroupsCore]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9e1a5e9f3b81ad6a5c613d181664a0efc6fe6dd7"
uuid = "d5909c97-4eac-4ecc-a3dc-fdd0858a4120"
version = "0.4.0"

[[deps.Gumbo]]
deps = ["AbstractTrees", "Gumbo_jll", "Libdl"]
git-tree-sha1 = "a1a138dfbf9df5bace489c7a9d5196d6afdfa140"
uuid = "708ec375-b3d6-5a57-a7ce-8257bf98657a"
version = "0.8.2"

[[deps.Gumbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "29070dee9df18d9565276d68a596854b1764aa38"
uuid = "528830af-5a63-567c-a44a-034ed33b8444"
version = "0.10.2+0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "abbbb9ec3afd783a7cbd82ef01dcd088ea051398"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImplicitDifferentiation]]
deps = ["AbstractDifferentiation", "Krylov", "LinearAlgebra", "LinearOperators", "PrecompileTools", "Requires", "SimpleUnPack"]
git-tree-sha1 = "d9f3708b9ccac5a9bf3dd99d010a6ac0b537eb83"
uuid = "57b37032-215b-411a-8a7c-41a003a55207"
version = "0.5.2"

    [deps.ImplicitDifferentiation.extensions]
    ImplicitDifferentiationChainRulesCoreExt = "ChainRulesCore"
    ImplicitDifferentiationForwardDiffExt = "ForwardDiff"
    ImplicitDifferentiationStaticArraysExt = "StaticArrays"
    ImplicitDifferentiationZygoteExt = "Zygote"

    [deps.ImplicitDifferentiation.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "68772f49f54b479fa88ace904f6127f0a3bb2e46"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.12"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "95220473901735a0f4df9d1ca5b171b568b2daa3"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.13.2"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "Printf", "SnoopPrecompile", "SparseArrays"]
git-tree-sha1 = "25b2fcda4d455b6f93ac753730d741340ba4a4fe"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.16.0"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "17e462054b42dcdda73e9a9ba0c67754170c88ae"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.9.4"

[[deps.LDLFactorizations]]
deps = ["AMD", "LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "70f582b446a1c3ad82cf87e62b878668beef9d13"
uuid = "40e66cde-538c-5869-a4ad-c39174c6795b"
version = "0.10.1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "LinearAlgebra", "MacroTools", "PreallocationTools", "RecursiveArrayTools", "StaticArrays"]
git-tree-sha1 = "cd04158424635efd05ff38d5f55843397b7416a9"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.14.0"

[[deps.LambertW]]
git-tree-sha1 = "c5ffc834de5d61d00d2b0e18c96267cffc21f648"
uuid = "984bce1d-4616-540c-a9ee-88d1112d94c9"
version = "0.4.6"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LatticeRules]]
deps = ["Random"]
git-tree-sha1 = "7f5b02258a3ca0221a6a9710b0a0a2e8fb4957fe"
uuid = "73f95e8e-ec14-4e6a-8b18-0d2e271c4e55"
version = "0.0.1"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "62edfee3211981241b57ff1cedf4d74d79519277"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.15"

[[deps.LazilyInitializedFields]]
git-tree-sha1 = "8f7f3cabab0fd1800699663533b6d5cb3fc0e612"
uuid = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
version = "1.2.2"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyStack]]
deps = ["ChainRulesCore", "LinearAlgebra", "NamedDims", "OffsetArrays"]
git-tree-sha1 = "2eb4a5bf2eb0519ebf40c797ba5637d327863637"
uuid = "1fad7336-0346-5a1a-a56f-a06ba010965b"
version = "0.0.8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9df2ab050ffefe870a09c7b6afdb0cde381703f2"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.1"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

[[deps.LinearOperators]]
deps = ["FastClosures", "LDLFactorizations", "LinearAlgebra", "Printf", "SparseArrays", "TimerOutputs"]
git-tree-sha1 = "a58ab1d18efa0bcf9f0868c6d387e4126dad3e72"
uuid = "5c8ed15e-5a4c-59e4-a42b-c7e8811fb125"
version = "2.5.2"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "0f5648fbae0d015e3abe5867bca2b362f67a5894"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.166"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MacroModelling]]
deps = ["AbstractDifferentiation", "AxisKeys", "BlockTriangularForm", "ChainRulesCore", "DataStructures", "DocStringExtensions", "DynarePreprocessor_jll", "ForwardDiff", "ImplicitDifferentiation", "JSON", "JuMP", "Krylov", "LaTeXStrings", "LinearAlgebra", "LinearOperators", "MacroTools", "MadNLP", "MatrixEquations", "NLopt", "PrecompileTools", "REPL", "Random", "RecursiveFactorization", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SparseArrays", "SpecialFunctions", "SpeedMapping", "Subscripts", "SymPyPythonCall", "Symbolics", "Unicode"]
git-tree-sha1 = "ed0c476fd865de68718f3cc14886288eca3d297b"
uuid = "687ffad2-3618-405e-ac50-e0f7b9c75e44"
version = "0.1.31"

    [deps.MacroModelling.weakdeps]
    StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
    Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.MadNLP]]
deps = ["Libdl", "LinearAlgebra", "Logging", "MathOptInterface", "NLPModels", "Pkg", "Printf", "SolverCore", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "65d1edba975973dfe3d08f06dde3b95847f8f233"
uuid = "2621e9c9-9eb4-46b1-8089-e8c72242dfb6"
version = "0.7.0"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MarkdownAST]]
deps = ["AbstractTrees", "Markdown"]
git-tree-sha1 = "465a70f0fc7d443a00dcdc3267a497397b8a3899"
uuid = "d0879d2d-cac2-40c8-9cee-1863dc0c7391"
version = "0.1.2"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "362ae34a5291a79e16b8eb87b5738532c5e799ff"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.23.0"

[[deps.MatrixEquations]]
deps = ["LinearAlgebra", "LinearMaps"]
git-tree-sha1 = "825264b8ce24f1ea5ea5f7572be8c4cf34973337"
uuid = "99c1a7ee-ab34-5fd5-8076-27c950a045f4"
version = "2.3.2"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MicroMamba]]
deps = ["Pkg", "Scratch", "micromamba_jll"]
git-tree-sha1 = "011cab361eae7bcd7d278f0a7a00ff9c69000c51"
uuid = "0b3b1443-0f03-428d-bdfb-f27f9c1191ea"
version = "0.1.14"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "6ffb234d6d7c866a75c1879d2099049d3a35a83a"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.3"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "806eea990fb41f9b36f1253e5697aa645bf6a9f8"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.0"

[[deps.NLPModels]]
deps = ["FastClosures", "LinearAlgebra", "LinearOperators", "Printf", "SparseArrays"]
git-tree-sha1 = "51b458add76a938917772ee661ffb9d59b4c7e5d"
uuid = "a4795742-8479-5a88-8948-cc11e1c8c1a6"
version = "0.20.0"

[[deps.NLopt]]
deps = ["NLopt_jll"]
git-tree-sha1 = "19d2a1c8a3c5b5a459f54a10e54de630c4a05701"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "1.0.0"
weakdeps = ["MathOptInterface"]

    [deps.NLopt.extensions]
    NLoptMathOptInterfaceExt = ["MathOptInterface"]

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9b1f15a08f9d00cdb2761dcfa6f453f5d0d6f973"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.1+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "dc9144f80a79b302b48c282ad29b1dc2f10a9792"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "1.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "2ac17d29c523ce1cd38e27785a7d23024853a4bb"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSH_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1b2f042897343a9dfdcc9366e4ecbd3d00780c49"
uuid = "9bd350c2-7e96-507f-8002-3f2e150b4e1b"
version = "8.9.0+1"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4e5be6bb265d33669f98eb55d2a57addd1eeb72c"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.30"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "fca25670784a1ae44546bcb17288218310af2778"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.9"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "Requires"]
git-tree-sha1 = "f739b1b3cc7b9949af3b35089931f2b58c289163"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.12"

    [deps.PreallocationTools.extensions]
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "1d05623b5952aed1307bf8b43bec8b8d1ef94b6e"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.PythonCall]]
deps = ["CondaPkg", "Dates", "Libdl", "MacroTools", "Markdown", "Pkg", "REPL", "Requires", "Serialization", "Tables", "UnsafePointers"]
git-tree-sha1 = "4999b3e4e9bdeba0b61ede19cc45a2128db21cdc"
uuid = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
version = "0.9.15"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.QuasiMonteCarlo]]
deps = ["Accessors", "ConcreteStructs", "LatticeRules", "LinearAlgebra", "Primes", "Random", "Requires", "Sobol", "StatsBase"]
git-tree-sha1 = "cc086f8485bce77b6187141e1413c3b55f9a4341"
uuid = "8a4e6c94-4038-4cdc-81c3-7e6ffdb2a71b"
version = "0.3.3"
weakdeps = ["Distributions"]

    [deps.QuasiMonteCarlo.extensions]
    QuasiMonteCarloDistributionsExt = "Distributions"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "b8a399e95663485820000f26b6a43c794e166a49"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.4"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "d7087c013e8a496ff396bae843b1e16d9a30ede8"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.38.10"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "PrecompileTools", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "8bc86c78c7d8e2a5fe559e3721c0f9c9e303b2ed"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.21"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegistryInstances]]
deps = ["LazilyInitializedFields", "Pkg", "TOML", "Tar"]
git-tree-sha1 = "ffd19052caf598b8653b99404058fce14828be51"
uuid = "2792f1a3-b283-48e8-9a74-f99dce5104f3"
version = "0.1.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "6aacc5eefe8415f47b3e34214c1d79d2674a0ba2"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.12"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d8911cc125da009051fb35322415641d02d9e37f"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.6"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "3aac6d68c5e57449f5b9b865c9ba50ac2970c4cf"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.42"

[[deps.Sass]]
deps = ["libsass_jll"]
git-tree-sha1 = "aa841c3738cec78b5dbccd56dda332710f35f6a5"
uuid = "322a6be2-4ae8-5d68-aaf1-3e960788d1d9"
version = "0.2.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FillArrays", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "QuasiMonteCarlo", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces"]
git-tree-sha1 = "d432b4c4cc922fb7b21b555c138aa87f9fb7beb8"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.9.1"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "Lazy", "LinearAlgebra", "Setfield", "SparseArrays", "StaticArraysCore", "Tricks"]
git-tree-sha1 = "51ae235ff058a64815e0a2c34b1db7578a06813d"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.7"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sobol]]
deps = ["DelimitedFiles", "Random"]
git-tree-sha1 = "5a74ac22a9daef23705f010f72c81d6925b19df8"
uuid = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
version = "1.5.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SolverCore]]
deps = ["LinearAlgebra", "NLPModels", "Printf"]
git-tree-sha1 = "9fb0712d597d6598857ae50b7744df17b1137b38"
uuid = "ff4d7338-4cf1-434d-91df-b86cb86fb843"
version = "0.3.7"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SpeedMapping]]
deps = ["AccurateArithmetic", "BenchmarkTools", "Coverage", "DocumenterTools", "ForwardDiff", "LinearAlgebra", "MuladdMacro", "SparseArrays"]
git-tree-sha1 = "f90e5469bcdb5d8c0c367b2bf7db0807fe670691"
uuid = "f1835b91-879b-4a3f-a438-e4baacf14412"
version = "0.3.0"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "03fec6800a986d191f64f5c0996b59ed526eda25"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.1"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "5ef59aea6f18c25168842bded46b16662141ab87"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.7.0"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "d6415f66f3d89c615929af907fdc6a3e17af0d8c"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.2"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.Subscripts]]
git-tree-sha1 = "1a7e74e19e1a430e8407902fd0a384b90f8415d3"
uuid = "2b7f82d5-8785-4f63-971e-f18ddbeb808e"
version = "0.1.2"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymPyCore]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "fa24a1be876b01aca1a47beff39a9648dbca38d2"
uuid = "458b697b-88f0-4a86-b56b-78b75cfb3531"
version = "0.1.7"
weakdeps = ["SymbolicUtils"]

    [deps.SymPyCore.extensions]
    SymPyCoreSymbolicUtilsExt = "SymbolicUtils"

[[deps.SymPyPythonCall]]
deps = ["CommonEq", "CommonSolve", "CondaPkg", "LinearAlgebra", "PythonCall", "SpecialFunctions", "SymPyCore"]
git-tree-sha1 = "1005d659221ce8dcf11061c7a37788d5bbd1f80f"
uuid = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
version = "0.2.4"
weakdeps = ["Symbolics"]

    [deps.SymPyPythonCall.extensions]
    SymPyPythonCallSymbolicsExt = "Symbolics"

[[deps.SymbolicIndexingInterface]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "f8ab052bfcbdb9b48fad2c80c873aa0d0344dfe5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.2.2"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TimerOutputs", "Unityper"]
git-tree-sha1 = "2f3fa844bcd33e40d8c29de5ee8dded7a0a70422"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "1.4.0"

[[deps.Symbolics]]
deps = ["ArrayInterface", "Bijections", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "Groebner", "IfElse", "LaTeXStrings", "LambertW", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "PrecompileTools", "RecipesBase", "RecursiveArrayTools", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TreeViews"]
git-tree-sha1 = "4d4e922e160827388c003a9a088a4c63f339f6c0"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "5.10.0"

    [deps.Symbolics.extensions]
    SymbolicsSymPyExt = "SymPy"

    [deps.Symbolics.weakdeps]
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "fadebab77bf3ae041f77346dd1c290173da5a443"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.1.20"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "21c8fc7cd598ef49f11bc9e94871f5d7740e34b9"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.5"

[[deps.UnsafePointers]]
git-tree-sha1 = "c81331b3b2e60a982be57c046ec91f599ede674a"
uuid = "e17b2a0c-0bdf-430a-bd0c-3a23cae4ff39"
version = "1.0.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "7209df901e6ed7489fe9b7aa3e46fb788e15db85"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.65"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libsass_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "941afb93587dcec07f89e511057f5efc0bec6f0d"
uuid = "47bcb7c8-5119-555a-9eeb-0afcc36cd728"
version = "3.6.4+0"

[[deps.micromamba_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "66d07957bcf7e4930d933195aed484078dd8cbb5"
uuid = "f8abcde7-e9b7-5caa-b8af-a437887ae8e4"
version = "1.4.9+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═d93e2dfc-a0be-4dba-8526-b6bd0301a49a
# ╠═0f0b1b84-7899-4013-8005-a52ac8664714
# ╠═3e1ebba0-9229-11ee-230d-251f23eaf3f4
# ╠═2de2b8b4-5d8e-4339-bdcc-9c5700dcba2c
# ╠═26613261-5fd5-46f4-993d-3442f9f0a6d1
# ╠═cdf1bbca-26f0-4987-823a-5fa951181143
# ╠═bfc386c1-c979-4e85-baee-ed07c1d7d82e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
