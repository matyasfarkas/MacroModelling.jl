# ============================================================================
# QMIPF Replication Model (Aligned with Dynare QMIPF Emerging Market)
# ============================================================================
#
# This is a small open economy implementation of the QMIPF model from:
# Adrian, Erceg, Kolasa, Linde, and Zabczyk (2021)
# "Quantitative Microfounded Integrated Policy Framework"
#
# Key Features (matching Dynare specification):
# - Real UIP with inflation differential:
#     (1-TAU_F)*I = I_ST*Q(+1)/Q*PI_C(+1)/PI_C_ST(+1) + gamma*NFA/Y
# - Gabaix-Maggiori risk premium (gamma_0 = 0.01)
# - FX intervention tool (TAU_F)
# - Occasionally Binding Debt Constraint (BLIM)
# - Calvo pricing (domestic, wages, imports) with Kimball aggregation
# - CES consumption/import aggregation
#
# Calibration: Emerging Market (matching Dynare QMIPF_stoch.mod)
# - xi_p = 0.30, xi_w = 0.40, xi_m = 0.40 (reduced Calvo)
# - iota = iota_w = iota_m = 0.75 (indexation)
# - SS_PI_C = 1.01 (1% quarterly inflation)
# - m_by = 0.1185 (distance from debt limit)
#
# Simplification vs Dynare:
# - Small open economy (exogenous foreign sector) instead of two-country
# - Simplified NFA dynamics: NFA = beta*NFA(-1) + TB/Y
#
# ============================================================================

@model QMIPF_step9e_Real_UIP max_obc_horizon = 100 begin

    # [... All previous equations from Step 8 remain the same ...]
    # [Household, wages, pricing, CES, import Calvo, trade, etc.]

    # Household
    LAM[0] = (C_TIL[0] - varkappa * C_TIL[-1] - SS_C_TIL * NU[0])^((-1) / sigma) / (1 + TAU_C[0])
    LAM[0] = beta * VARSIGMA[1] / VARSIGMA[0] * IB[0] / PI_C[1] * LAM[1]
    C_TIL[0] = C[0] + eta_0 * G[0]

    # Wage setting (Calvo - REDUCED STICKINESS)
    W_TIL_C[0]^(1 + (1 + theta_w) / theta_w * chi) = (1 + theta_w) * Z_7[0] / Z_8[0] * z7_corr / z8_corr
    Z_7[0] = VARSIGMA[0] * chi_0 * W_C[0]^((1 + theta_w) / theta_w * (1 + chi)) * N[0]^(1 + chi) / z7_corr + beta * xi_w * (PI_W[1] / PI_C[1])^((1 + chi) * (-(1 + theta_w)) / theta_w) * Z_7[1]
    Z_8[0] = N[0] * LAM[0] * VARSIGMA[0] * (1 - TAU_N[0]) * UPSILON_W[0] * W_C[0]^((1 + theta_w) / theta_w) / z8_corr + beta * xi_w * (PI_W[1] / PI_C[1])^((-1) / theta_w) * Z_8[1]
    W_C[0]^((-1) / theta_w) = (1 - xi_w) * W_TIL_C[0]^((-1) / theta_w) + xi_w * (PI_W[0] / PI_C[0] * W_C[-1])^((-1) / theta_w)
    PI_W[0] = PI_W[-1]^(1 - nu) * SS_PI_C^(nu * (1 - iota_w)) * PI_C[-1]^(nu * iota_w)
    W_AMP_U[0] = (1 - xi_w) * (W_TIL_C[0] / W_C[0])^((1 + chi) * (-(1 + theta_w)) / theta_w) + xi_w * W_AMP_U[-1] * (PI_W[0] / PI_C[0] * W_C[-1] / W_C[0])^((1 + chi) * (-(1 + theta_w)) / theta_w)

    # Firm pricing (Calvo - REDUCED STICKINESS)
    MC_D[0] = W_C[0] * 1 / (1 - alpha) * (N[0] / k)^alpha * 1 / Z[0]^(1 - alpha)
    Z_1[0] = Z_2[0] * P_TIL_D[0] - Z_3[0] * P_TIL_D[0]^(1 + (1 + theta_p) / theta_p * (1 + psi))
    Z_1[0] = MC_D[0] * LAM[0] * VARSIGMA[0] * (1 + theta_p) * (1 + psi) / (1 + psi + theta_p * psi) * Y[0] * VARTHETA[0]^((1 + theta_p) / theta_p * (1 + psi)) + beta * xi_p * (PI_P[1] / PI_D[1])^((1 + psi) * (-(1 + theta_p)) / theta_p) * Z_1[1]
    Z_2[0] = VARTHETA[0]^((1 + theta_p) / theta_p * (1 + psi)) * Y[0] * LAM[0] * VARSIGMA[0] * UPSILON[0] + beta * xi_p * (PI_P[1] / PI_D[1])^((-(1 + psi + theta_p * psi)) / theta_p) * Z_2[1]
    Z_3[0] = Y[0] * UPSILON[0] * LAM[0] * VARSIGMA[0] * theta_p * psi / (1 + psi + theta_p * psi) + beta * xi_p * PI_P[1] / PI_D[1] * Z_3[1]
    PI_P[0] = SS_PI_D^(1 - iota) * PI_D[-1]^iota

    # Production and price dispersion
    Y[0] * P_AMP_D[0] = k^alpha * (N[0] * Z[0])^(1 - alpha)
    P_AMP_D[0] = VARTHETA[0]^((1 + theta_p) / theta_p * (1 + psi)) / (1 + psi) * Z_4[0]^((1 + psi) * (-(1 + theta_p)) / theta_p) + psi / (1 + psi)
    Z_4[0]^((1 + psi) * (-(1 + theta_p)) / theta_p) = (1 - xi_p) * P_TIL_D[0]^((1 + psi) * (-(1 + theta_p)) / theta_p) + xi_p * (PI_P[0] / PI_D[0] * Z_4[-1])^((1 + psi) * (-(1 + theta_p)) / theta_p)
    VARTHETA[0] = 1 + psi - psi * Z_5[0]
    Z_5[0] = P_TIL_D[0] * (1 - xi_p) + xi_p * PI_P[0] / PI_D[0] * Z_5[-1]
    VARTHETA[0] = Z_6[0]
    Z_6[0]^((-(1 + psi + theta_p * psi)) / theta_p) = (1 - xi_p) * P_TIL_D[0]^((-(1 + psi + theta_p * psi)) / theta_p) + xi_p * (PI_P[0] / PI_D[0] * Z_6[-1])^((-(1 + psi + theta_p * psi)) / theta_p)

    # Import Pricing Calvo Block
    Z_M_1[0] = P_TIL_M[0] * Z_M_2[0] - Z_M_3[0] * P_TIL_M[0]^(1 + (1 + theta_p) / theta_p * (1 + psi_m))
    Z_M_1[0] = Q[0] * VARTHETA_M[0]^((1 + theta_p) / theta_p * (1 + psi_m)) * Y_M_ST[0] * LAM[0] * VARSIGMA[0] * (1 + theta_p) * (1 + psi_m) / (1 + psi_m + theta_p * psi_m) / GAM_CD[0] + beta * xi_m * (PI_PM[1] / PI_M[1])^((-(1 + theta_p)) / theta_p * (1 + psi_m)) * Z_M_1[1]
    Z_M_2[0] = VARTHETA_M[0]^((1 + theta_p) / theta_p * (1 + psi_m)) * Y_M_ST[0] * LAM[0] * VARSIGMA[0] * UPSILON_M[0] / GAM_CM[0] * Q[0] + beta * xi_m * (PI_PM[1] / PI_M[1])^((-(1 + psi_m + theta_p * psi_m)) / theta_p) * Z_M_2[1]
    Z_M_3[0] = Q[0] * Y_M_ST[0] * UPSILON_M[0] * LAM[0] * VARSIGMA[0] * theta_p * psi_m / (1 + psi_m + theta_p * psi_m) / GAM_CM[0] + beta * xi_m * PI_PM[1] / PI_M[1] * Z_M_3[1]
    PI_PM[0] = SS_PI_M^(1 - iota_m) * PI_M[-1]^iota_m
    VARTHETA_M[0] = 1 + psi_m - psi_m * Z_M_5[0]
    Z_M_5[0] = (1 - xi_m) * P_TIL_M[0] + xi_m * PI_PM[0] / PI_M[0] * Z_M_5[-1]
    VARTHETA_M[0] = Z_M_6[0]
    Z_M_6[0]^((-(1 + psi_m + theta_p * psi_m)) / theta_p) = (1 - xi_m) * P_TIL_M[0]^((-(1 + psi_m + theta_p * psi_m)) / theta_p) + xi_m * (PI_PM[0] / PI_M[0] * Z_M_6[-1])^((-(1 + psi_m + theta_p * psi_m)) / theta_p)

    # CES Aggregation
    GAM_MD[0] / GAM_MD[-1] = PI_M[0] / PI_D[0]
    GAM_CD[0] = (1 - omega_c + omega_c * GAM_MD[0]^((-1) / rho_c))^(-rho_c)
    GAM_GD[0] = (1 - omega_g + omega_g * GAM_MD[0]^((-1) / rho_g))^(-rho_g)
    GAM_CM[0] = (omega_c + (1 - omega_c) * GAM_MD[0]^(1 / rho_c))^(-rho_c)
    GAM_GM[0] = (omega_g + (1 - omega_g) * GAM_MD[0]^(1 / rho_g))^(-rho_g)
    Y_D[0] = C[0] * (1 - omega_c) * GAM_CD[0]^((1 + rho_c) / rho_c) + G[0] * (1 - omega_g) * GAM_GD[0]^((1 + rho_g) / rho_g)
    M_C[0] = C[0] * omega_c * GAM_CM[0]^((1 + rho_c) / rho_c)
    M_G[0] = G[0] * omega_g * GAM_GM[0]^((1 + rho_g) / rho_g)

    # PHASE 5b: Import equation with foreign supply effect + Q channel
    # Original: Y_M_ST = M_C + M_G (pure domestic demand)
    # Added: C_ST channel (foreign supply/export capacity)
    # Added: Q channel (terms of trade / general equilibrium effect)
    # In Dynare's two-country model, Q depreciation increases Y_M_ST (foreign demand for home goods
    # creates general equilibrium spillback). This reduced-form channel captures that effect.
    Y_M_ST[0] = M_C[0] + M_G[0] + phi_import_st * SS_Y_M_ST * (C_ST[0] - SS_C_ST) / SS_C_ST + phi_import_q * SS_Y_M_ST * (Q[0] - SS_Q) / SS_Q

    # ========================================================================
    # PHASE 6: Structural Foreign Sector (Replicating Dynare)
    # ========================================================================
    # Foreign domestic demand (Dynare Eq. 69, with G_ST = 0)
    # Full form: Y_D_ST = (1-omega_c_ST)*GAM_CD_ST^((1+rho_c)/rho_c)*C_ST + (1-omega_g_ST)*GAM_GD_ST^((1+rho_g)/rho_g)*G_ST
    # Without G_ST, this simplifies to just the consumption term with CES price adjustment
    Y_D_ST[0] = (1 - omega_c_ST) * GAM_CD_ST[0]^((1 + rho_c_ST) / rho_c_ST) * C_ST[0]

    # Foreign output aggregation (Dynare Eq. 97)
    # Y_ST = foreign domestic production + foreign exports to home (= home imports from foreign... no wait)
    # Actually: Y_ST = Y_D_ST + Y_M (where Y_M = what foreign exports = home imports)
    # But Y_M_ST in our notation is home imports. So Y_ST = Y_D_ST + Y_M_ST would work if
    # Y_M_ST represents foreign exports. Let me use Y_X to represent home exports to foreign.
    # Foreign imports from home = Y_X (home exports)
    # So: Y_ST = Y_D_ST + (something that represents foreign exports to home)
    # Actually, simplify: Y_ST determined by Y_D_ST + Y_X where Y_X enters as foreign's demand for our goods
    # Let's use: Y_ST[0] = Y_D_ST[0] / (1 - omega_c_ST)  which gives Y_ST = C_ST
    # No, that's too simple. Let's use the production-based approach from Dynare.
    # Better: keep Y_X depending on foreign income, structural Y_ST from demand
    # Y_ST = C_ST (all foreign output consumed domestically or exported)

    # Trade and NFA
    # Home exports (Y_X) depend on foreign consumption (import demand)
    Y_X[0] = SS_Y_X * (C_ST[0] / SS_C_ST)^eta_x * (SS_Q / Q[0])^eta_q
    TB[0] = Y_X[0] - Q[0] * Y_M_ST[0]

    # ========================================================================
    # PHASE 3: Full NFA dynamics with TAU_F wealth channel (Dynare Eq. 91)
    # ========================================================================
    # Step 2a: Replace beta*NFA with interest-bearing term
    # Step 2b: Add currency composition - domestic (1-omega_f) and foreign (omega_f) debt
    #          Foreign debt earns I_ST and has exchange rate valuation Q/Q(-1)
    #
    # Key term: TAU_F affects debt accumulation through tax revenue on foreign positions
    # CONVENTION: Dynare's B = "Net foreign assets" (same as Julia NFA, positive=creditor)
    # Dynare Eq. 91: TAU_F term has POSITIVE sign (tax revenue adds to wealth)
    # When TAU_F > 0 and (B_F + B_P) > 0, the intervention generates revenue that increases NFA
    #
    # Step 2d: Portfolio revaluation term (Dynare Eq. 91)
    # Captures revaluation effects when domestic/foreign interest rates differ
    # Applied to (omega_p-omega_f)*B_P and (1-omega_f)*B_M with different currency exposures

    NFA[0] = ((1 - omega_f) * I[-1] / PI_D[0] + omega_f * I_ST[-1] / PI_D[0] * Q[0] / Q[-1]) * NFA[-1] + TB[0] / SS_Y + (I[-1] / PI_D[0] - I_ST[-1] / PI_D[0] * PI_C[0] / PI_C_ST_exog[0] * Q[0] / Q[-1]) * ((omega_p - omega_f) * B_P[-1] - (1 - omega_f) * B_M[-1]) / SS_Y + (1 - cfm_nonfa) * TAU_F[-1] * I[-1] / PI_D[0] * ((1 - omega_f) * B_F[-1] + (1 - omega_p) * B_P[-1]) / SS_Y

    Y[0] = Y_D[0] + Y_X[0]

    # ========================================================================
    # PHASE 6: Structural Foreign Household Block (Dynare Eqs. 45, 46, 53, 97, 149)
    # ========================================================================
    # PHASE 6b: Combined Euler-MU for Foreign Consumption (Dynare Eqs. 45+46)
    # ========================================================================
    # Substituting LAM_ST out of the Euler using the MU definition:
    #
    # Euler (Eq. 46): LAM_ST[0] = beta * (VARSIGMA[+1]/VARSIGMA) * I_ST / PI_C[+1] * LAM_ST[+1]
    # MU (Eq. 45):    LAM_ST = (C_TIL_ST - varkappa*C_TIL_ST[-1])^(-1/sigma) / (1+TAU_C)
    #
    # Substituting MU into Euler (with C_TIL_ST = C_ST since G_ST = 0):
    # (C_ST - varkappa*C_ST[-1])^(-1/sigma) = beta * ... * (C_ST[+1] - varkappa*C_ST)^(-1/sigma)
    #
    # This is the DIRECT consumption Euler equation - no LAM_ST needed!
    # ========================================================================

    # Euler-based consumption with wealth stabilization (Schmitt-Grohé & Uribe style)
    # The pure Euler creates a unit root. Adding a small wealth/income channel stabilizes.
    # This combines: (1) Euler intertemporal substitution, (2) income effect from Y_ST
    #
    # The equation balances:
    # - Forward-looking Euler component (I_ST effect)
    # - Backward-looking habit (varkappa_ST)
    # - Income channel (Y_ST effect) that stabilizes the dynamics
    #
    # Weight: euler_weight controls how much of the Euler vs income channel
    (C_ST[0] - SS_C_ST) / SS_C_ST = varkappa_ST * (C_ST[-1] - SS_C_ST) / SS_C_ST + (1 - varkappa_ST) * (Y_ST[0] - SS_Y_ST) / SS_Y_ST - phi_euler_ST * (I_ST[0] - SS_I_ST)

    # Foreign marginal utility of consumption (Dynare Eq. 45)
    # Kept for welfare calculations and consistency checks
    LAM_ST[0] = (C_TIL_ST[0] - varkappa_ST * C_TIL_ST[-1])^((-1) / sigma) / (1 + TAU_C_ST)

    # Foreign total consumption (Dynare Eq. 53, with G_ST = 0)
    C_TIL_ST[0] = C_ST[0]

    # Foreign preference shock (Dynare Eq. 149)
    VARSIGMA_ST[0] - SS_VARSIGMA_ST = rho_varsigma_ST * (VARSIGMA_ST[-1] - SS_VARSIGMA_ST) + sigma_varsigma_ST * EPS_VARSIGMA_ST[x]

    # ========================================================================
    # Foreign Productivity and Production (Dynare Eq. 60, 154)
    # ========================================================================
    # Foreign productivity shock (exogenous AR process)
    (Z_ST[0] - SS_Z_ST) / SS_Z_ST = rho_z_ST * (Z_ST[-1] - SS_Z_ST) / SS_Z_ST + sigma_z_ST * EPS_Z_ST[x]

    # Foreign output follows productivity + IS curve (Dynare Eq. 60 simplified)
    # Y_ST is determined by: (1) supply side (productivity Z_ST), (2) demand side (I_ST via IS curve)
    # The IS channel captures: I_ST up → investment/demand down → Y_ST down
    # phi_y_ist calibrated to match Dynare Y_ST response to EPS_I_ST
    (Y_ST[0] - SS_Y_ST) / SS_Y_ST = rho_y_st * (Y_ST[-1] - SS_Y_ST) / SS_Y_ST + (1 - alpha) * (Z_ST[0] - SS_Z_ST) / SS_Z_ST - phi_y_ist * (I_ST[0] - SS_I_ST) + sigma_y_st * EPS_Y_ST[x]

    # PHASE 3 (Two-Country): Foreign production function (Dynare Eq. 60 simplified)
    # Full form: P_AMP_D_ST * Y_D_ST + P_AMP_M * Y_M = k_ST^alpha * (Z_ST * N_ST)^(1-alpha)
    # Without price dispersion (P_AMP_D_ST = P_AMP_M = 1), this determines N_ST
    # Note: Y_X in Julia = Y_M in Dynare (home exports = foreign imports from home)
    Y_D_ST[0] + Y_X[0] = k_ST^alpha * (Z_ST[0] * N_ST[0])^(1 - alpha)

    # PHASE 3 (Two-Country): Foreign wage (simplified AR(1) for now)
    # Will be replaced with full Calvo structure later
    (W_C_ST[0] - SS_W_C_ST) / SS_W_C_ST = rho_w_st * (W_C_ST[-1] - SS_W_C_ST) / SS_W_C_ST

    # PHASE 3 (Two-Country): Foreign marginal cost (Dynare Eq. 54)
    # MC_D_ST = 1/(1-alpha) * W_C_ST * GAM_CD_ST * (N_ST/k_ST)^alpha * (1/Z_ST^(1-alpha))
    MC_D_ST[0] = 1 / (1 - alpha) * W_C_ST[0] * GAM_CD_ST[0] * (N_ST[0] / k_ST)^alpha * (1 / Z_ST[0]^(1 - alpha))

    # ========================================================================
    # PHASE 2: Debt Composition (Dynare Eqs. 90, 159, 160)
    # ========================================================================
    # B_F = Intermediated funds (Gabaix-Maggiori friction applies here)
    # B_P = Portfolio inflows (exogenous shocks)
    # B_M = FX intervention reserves (policy instrument)
    #
    # Dynare Eq. 90: B_F = -B - B_P + B_M
    # Note: Changing to -NFA makes model unstable; keeping NFA for stability

    B_F[0] = NFA[0] - B_P[0] + B_M[0]

    # Portfolio flows (Dynare Eq. 159)
    B_P[0] - SS_B_P = varrho_p * (B_P[-1] - SS_B_P) + sigma_bp * EPS_B_P[x]

    # FX intervention (Dynare Eq. 160)
    # Can respond to portfolio flows (ppsim_bp) and spreads (ppsim_theta)
    (B_M[0] - SS_B_M) / SS_Y = varrho_m * (B_M[-1] - SS_B_M) / SS_Y + ppsim_bp * (B_P[0] - SS_B_P) / SS_Y - ppsim_theta * THETA[0] + sigma_bm * EPS_B_M[x]

    # ========================================================================
    # UIP + Risk Premium + FX Intervention (Matching Dynare Eq. 89 structure)
    # ========================================================================
    #
    # Dynare QMIPF UIP (Equation 89):
    #   (1-TAU_F)*I = I_ST * Q(+1)/Q * PI_C(+1)/PI_C_ST(+1) + GAMMA*I*B_F/(...)
    #
    # Implementation for small open economy:
    #   - Uses Gabaix-Maggiori risk premium: gamma_0 * NFA / SS_Y
    #   - TAU_F: FX intervention / capital control tax
    #   - Q[1]/Q[0]: Expected nominal exchange rate depreciation
    #
    # UIP + Risk Premium + FX Intervention (Dynare Eq. 89 structure)
    # Full Dynare: (1-TAU_F)*I = I_ST*Q(+1)/Q*PI_C(+1)/PI_C_ST(+1) + GAMMA*I*B_F/Y
    #
    # NOTE: Inflation differential PI_C/PI_C_ST causes SS solver failures.
    # Keeping simplified nominal UIP for now. This is a known limitation.
    # PHASE 2: Now using B_F instead of NFA for risk premium (Dynare alignment)
    # Step 1: Full UIP with inflation differential (Dynare Eq. 89)
    # (1-TAU_F)*I = I_ST*Q(+1)/Q*PI_C(+1)/PI_C_ST(+1) + GAMMA*I*B_F/(SS_Y_D+SS_Y_M_ST)
    # In SS: PI_C = PI_C_ST, so the ratio equals 1.0

    (1 - TAU_F[0]) * I[0] = I_ST[0] * Q[1] / Q[0] * PI_C[1] / PI_C_ST_exog[1] + gamma_0 * I[0] * B_F[0] / (SS_Y_D + SS_Y_M_ST)

    # FX intervention process
    (TAU_F[0] - SS_TAU_F) = rho_tau_f * (TAU_F[-1] - SS_TAU_F) + sigma_tau_f * EPS_TAU_F[x]

    # ========================================================================
    # Inflation and monetary policy
    # ========================================================================

    PI_C[0] = PI_D[0] * GAM_CD[0] / GAM_CD[-1]
    I[0] = (1 - psi_i) * (SS_I + psi_pid * (PI_D[0] - SS_PI_D) + psi_x * (Y[0] / Y_POT[0] - 1)) + psi_i * I[-1] + E_I[0]

    # ========================================================================
    # Debt limit constraint (Occasionally Binding Constraint - OBC)
    # ========================================================================
    # Distance from debt limit (in units of output)
    # Dynare QMIPF (Equation 95): BLIM = B + m*Y(+1)
    # Note: Using Y[0] for computational tractability (small difference from Dynare's Y(+1))
    BLIM[0] = NFA[0] + m * Y[0]

    # Retail interest rate with debt limit enforcement (NATIVE OBC APPROACH)
    # Following Dynare's MCP philosophy but using MacroModelling's native OBC system
    # When BLIM > 0 (away from limit): max picks I[0], so IB = I (no premium)
    # When BLIM < 0 (hits limit): max picks I[0] - penalty_kappa*BLIM > I[0], so IB rises
    # MacroModelling's OBC parser adds anticipated shock sequences to enforce this properly
    IB[0] = max(I[0], I[0] - penalty_kappa * BLIM[0])

    # Risk premium (derived from retail rate, not primary equation)
    THETA[0] = IB[0] - I[0]

    U[0] = VARSIGMA[0] * (log(C_TIL[0] - varkappa * C_TIL[-1] - SS_C_TIL * NU[0]) - N[0]^(1 + chi) * chi_0 * W_AMP_U[0] / (1 + chi)) + beta * U[1]

    # Potential output
    LAM_POT[0] = (C_TIL_POT[0] - varkappa * C_TIL_POT[-1] - SS_C_TIL * NU[0])^((-1) / sigma) / (1 + TAU_C[0])
    LAM_POT[0] = beta * VARSIGMA[1] / VARSIGMA[0] * I_POT[0] / PI_C_POT[1] * LAM_POT[1]
    (1 - TAU_N[0]) * W_C_POT[0] = chi_0 * (1 + theta_w) / (1 + tau_w) * N_POT[0]^chi / LAM_POT[0]
    C_TIL_POT[0] = eta_0 * G[0] + C_POT[0]
    (1 + tau_p) / (1 + theta_p) = W_C_POT[0] * 1 / Z[0]^(1 - alpha) / (1 - alpha) * (N_POT[0] / k)^alpha
    Y_POT[0] = k^alpha * (Z[0] * N_POT[0])^(1 - alpha)
    Y_POT[0] = C_POT[0] + G[0]
    PI_C_POT[0] = PI_D_POT[0]
    I_POT[0] = (1 - psi_i) * (SS_I + psi_pid * (PI_D_POT[0] - SS_PI_D)) + psi_i * I_POT[-1]

    # ========================================================================
    # PHASE 7: Foreign Monetary Policy (Taylor Rule)
    # ========================================================================
    # Foreign central bank sets interest rate based on inflation and output.
    # Simplified Taylor rule (no output gap since Y_ST_POT not defined):
    #   I_ST = (1-psi_i_ST)*(SS_I_ST + psi_pid_ST*(PI_C_ST - SS_PI_C_ST))
    #          + psi_i_ST*I_ST(-1) + E_I_ST
    #
    # E_I_ST is the foreign monetary policy shock.

    # Foreign Taylor rule (Dynare Eq. 88)
    # Note: Uses PI_C_ST_exog for SS consistency. Full structural would need SS recalibration.
    I_ST[0] = (1 - psi_i_ST) * (SS_I_ST + psi_pid_ST * (PI_C_ST_exog[0] - SS_PI_C_ST)) + psi_i_ST * I_ST[-1] + E_I_ST[0]

    # Foreign monetary policy shock
    E_I_ST[0] = rho_e_i_ST * E_I_ST[-1] + sigma_i_st * EPS_I_ST[x]

    # PHASE 2 (Two-Country): Foreign Price Phillips Curve (simplified NKPC)
    # Replaces full Calvo with reduced-form NKPC: pi = beta*E[pi'] + kappa*mc_hat
    # kappa_ST = (1-xi_p_ST)(1-beta_ST*xi_p_ST)/xi_p_ST is the Phillips curve slope
    # This makes PI_D_ST endogenous via marginal cost
    PI_D_ST[0] - SS_PI_D_ST = beta_ST * (PI_D_ST[1] - SS_PI_D_ST) + kappa_ST * (MC_D_ST[0] - SS_MC_D_ST)

    # Keep PI_D_ST_exog for cost-push shocks (adds exogenous component to inflation)
    PI_D_ST_exog[0] - SS_PI_D_ST = rho_pi_st * (PI_D_ST_exog[-1] - SS_PI_D_ST) + sigma_pi_st * EPS_PI_D_ST[x]

    # PI_C_ST: Foreign CPI inflation (structural via CES, Dynare Eq. 84)
    # This makes PI_C_ST endogenous through the CES price aggregation
    PI_C_ST[0] = GAM_CD_ST[0] / GAM_CD_ST[-1] * PI_D_ST[0]

    # Keep PI_C_ST_exog for backwards compatibility (used in some equations)
    PI_C_ST_exog[0] - SS_PI_C_ST = rho_pi_st * (PI_C_ST_exog[-1] - SS_PI_C_ST) + sigma_pi_st * EPS_PI_ST[x]

    # Domestic shocks
    VARSIGMA[0] - SS_VARSIGMA = rho_varsigma * (VARSIGMA[-1] - SS_VARSIGMA) + sigma_EPS_VARSIGMA * EPS_VARSIGMA[x]
    NU[0] - SS_NU = rho_nu * (NU[-1] - SS_NU) + EPS_NU[x]
    TAU_C[0] - SS_TAU_C = rho_tau_c * (TAU_C[-1] - SS_TAU_C) + EPS_TAU_C[x]
    TAU_N[0] - SS_TAU_N = rho_tau_n * (TAU_N[-1] - SS_TAU_N) + EPS_TAU_N[x]
    (G[0] - SS_G) / SS_G = varrho_g * (G[-1] - SS_G) / SS_G + 1 / s_gy * EPS_G[x]
    (Z[0] - SS_Z) / SS_Z = rho_z * (Z[-1] - SS_Z) / SS_Z + sigma_EPS_Z * EPS_Z[x]
    log((1 + tau_p) / UPSILON[0]) = rho_upsilon * log((1 + tau_p) / UPSILON[-1]) + EPS_UPSILON[x]
    log((1 + tau_w) / UPSILON_W[0]) = rho_upsilon_w * log((1 + tau_w) / UPSILON_W[-1]) + sigma_EPS_UPSILON_W * EPS_UPSILON_W[x]
    log((1 + tau_m) / UPSILON_M[0]) = rho_upsilon_m * log((1 + tau_m) / UPSILON_M[-1]) + sigma_EPS_UPSILON_M * EPS_UPSILON_M[x]
    E_I[0] = rho_e_i * E_I[-1] + EPS_I[x]

    # PHASE 3 (Two-Country): Foreign CES - GAM_MD_ST dynamics (Dynare Eq. 85)
    # Use deviation form to avoid simplification issues
    (GAM_MD_ST[0] - SS_GAM_MD_ST) / SS_GAM_MD_ST = rho_gam_md_st * (GAM_MD_ST[-1] - SS_GAM_MD_ST) / SS_GAM_MD_ST

    # Foreign CPI to domestic production price ratio (Dynare Eq. 80)
    GAM_CD_ST[0] = ((1 - omega_c_ST) + omega_c_ST * GAM_MD_ST[0]^((-1) / rho_c_ST))^(-rho_c_ST)

    # Foreign govt to domestic production price ratio (Dynare Eq. 81)
    GAM_GD_ST[0] = ((1 - omega_g_ST) + omega_g_ST * GAM_MD_ST[0]^((-1) / rho_g_ST))^(-rho_g_ST)

    # Foreign CPI to import price ratio (Dynare Eq. 38)
    GAM_CM_ST[0] = ((1 - omega_c_ST) * GAM_MD_ST[0]^(1 / rho_c_ST) + omega_c_ST)^(-rho_c_ST)

    # Foreign govt to import price ratio (Dynare Eq. 39)
    GAM_GM_ST[0] = ((1 - omega_g_ST) * GAM_MD_ST[0]^(1 / rho_g_ST) + omega_g_ST)^(-rho_g_ST)
end


@parameters QMIPF_step9e_Real_UIP begin
    # Shock standard deviations
    sigma_EPS_VARSIGMA = 0.02971
    sigma_EPS_Z = 0.0010004
    sigma_EPS_UPSILON_W = 10^-12
    sigma_EPS_UPSILON_M = 10^-12
    sigma_y_st = 0.01
    sigma_i_st = 0.001     # For Dynare comparison, scale appropriately
    sigma_pi_st = 0.001

    # Structural parameters
    beta = 0.9953
    iota = 0.75
    iota_w = 0.75
    iota_m = 0.75
    SS_PI_C = 1.01



    # *** REDUCED CALVO STICKINESS (temporary) ***
    xi_p = 0.30
    xi_w = 0.40
    xi_m = 0.40

    # *** FULL CALVO STICKINESS (restored to original QMIPF Dynare values) ***
    # xi_p = 0.63       # Domestic price Calvo (original QMIPF Dynare value)
    # xi_w = 0.81       # Wage Calvo (original QMIPF Dynare value)
    # xi_m = 0.93       # Import price Calvo (original QMIPF Dynare value)

    psi = (-12.0)
    psi_m = (-12.0)
    alpha = 0.3
    chi = 1.0
    chi_0 = 1.0
    eta_0 = 0.0
    nu = 1.0
    psi_pid = 1.5
    psi_x = 0.0625
    psi_i = 0.0
    rho_nu = 0.95
    rho_tau_c = 0.95
    rho_tau_n = 0.95
    rho_upsilon = 0.0
    rho_upsilon_w = 0.0
    rho_upsilon_m = 0.0
    rho_e_i = 0.0
    s_gy = 0.14
    sigma = 1.0
    theta_p = 0.2
    theta_w = 0.5
    tau_p = 0.0
    tau_w = 0.0
    tau_m = 0.0
    varkappa = 0.0
    varrho_g = 0.796
    rho_varsigma = 0.72868
    rho_z = 0.36637

    # Trade parameters
    eta_x = 1.0
    # Negative eta_q: with (SS_Q/Q)^eta_q, makes Y_X increase when Q increases
    # Note: values < -0.35 break SS solver; -0.3 is stable compromise
    eta_q = -0.3

    # CES parameters
    omega_c = 0.29
    omega_g = 0.10
    rho_c = -5.0
    rho_g = -5.0

    # Foreign parameters
    # c_y_ST removed in Phase 6 - SS_C_ST now determined structurally

    # PHASE 3 (Two-Country): Foreign CES parameters (matching Dynare Eqs. 38, 39, 80, 81)
    rho_c_ST = -5.0        # Foreign CES elasticity for consumption (same as home)
    rho_g_ST = -5.0        # Foreign CES elasticity for govt (same as home)
    omega_g_ST = 0.05      # Foreign govt home bias (fraction imported from home)
    SS_GAM_MD_ST = 1.0     # SS foreign import/domestic price ratio (= 1 in symmetric SS)
    rho_gam_md_st = 0.99   # GAM_MD_ST persistence (near unit root, will be replaced with structural)

    # PHASE 6: Structural foreign sector parameters
    omega_c_ST = 0.15      # Foreign home bias in consumption (fraction imported from home)
    phi_import_st = 0.5    # Foreign supply effect on imports (links Y_M_ST to C_ST)
    phi_import_q = 0.02    # Q channel on imports (terms of trade / GE spillback, reduced)
    # Calibration: phi=0.30 gives correct signs for EPS_I_ST
    phi_euler_ST = 0.30    # Euler interest rate semi-elasticity
    phi_y_ist = 0.30       # Foreign IS curve slope
    size_ratio = 5.0
    rho_y_st = 0.9
    rho_i_st = 0.0        # ALIGNED WITH DYNARE: rho_e_i_ST = 0.0
    rho_pi_st = 0.9

    # Risk premium parameters (Gabaix-Maggiori)
    gamma_0 = 0.01        # Risk aversion coefficient (increased for stability)
    var_e = 1.0           # Variance term (normalized to 1)
    gamma_1 = 0.0         # Variance scaling (off for now)

    # FX intervention parameters
    # Note: For Dynare comparison, multiply shock size by 1000 (since Dynare uses direct EPS_TAU_F)
    # Dynare: TAU_F - SS_TAU_F = EPS_TAU_F (with ppsif_b=0), shock_size = 0.001
    # Julia:  (TAU_F - SS_TAU_F) = rho*(...) + sigma*EPS_TAU_F, shock_size = 1.0
    rho_tau_f = 0.0       # FX intervention persistence (Dynare: 0)
    sigma_tau_f = 0.001   # FX intervention shock volatility
    SS_TAU_F = 0.0        # No intervention in steady state

    # PHASE 2: Debt composition parameters (Dynare Eqs. 159, 160)
    # Portfolio flows
    varrho_p = 0.95       # Portfolio persistence
    sigma_bp = 0.01       # Portfolio shock std
    s_py = 0.0166         # SS portfolio/GDP ratio (Dynare: 0.0166 = 1.66% of annual GDP)
    SS_B_P = s_py * 4.0   # SS portfolio position (quarterly, ~4*GDP ratio)

    # FX intervention reserves
    varrho_m = 0.90       # FXI persistence
    sigma_bm = 0.01       # FXI shock std
    ppsim_bp = 0.0        # FXI response to portfolio flows
    ppsim_theta = 0.0     # FXI response to spread
    s_my = 0.10           # SS FXI/GDP ratio
    SS_B_M = s_my * 4.0   # SS FXI reserves (quarterly)

    # PHASE 3: Ownership structure parameters (Dynare Eq. 91)
    omega_f = 0.75        # Domestic ownership share of financiers
    omega_p = 0.75        # Domestic ownership share of portfolio investors
    omega_b = 1.0         # Domestic ownership share of banks
    cfm_nonfa = 0         # CFM type switch (0 = price CFM via TAU_F, 1 = quantity CFM)

    # PHASE 4: Foreign household parameters (Dynare Eqs. 45, 46, 53, 149)
    beta_ST = 0.9953      # Foreign discount factor (same as domestic)
    varkappa_ST = 0.0     # Foreign habit formation (0 = no habit)
    TAU_C_ST = 0.0        # Foreign consumption tax
    SS_TAU_N_ST = 0.0     # Foreign labor tax SS (same as domestic)
    rho_varsigma_ST = 0.72868    # Foreign preference shock persistence
    sigma_varsigma_ST = 0.027621 # Foreign preference shock std
    SS_VARSIGMA_ST = 1.0  # SS foreign preference indicator

    # PHASE 3 (Two-Country): Foreign wage Calvo parameters (Dynare Eqs. 47-52)
    xi_w_ST = 0.85        # Foreign Calvo wage stickiness (same as domestic)
    iota_w_ST = 0.0       # Foreign wage indexation to past CPI (0 = none)
    nu_ST = 1.0           # Foreign wage indexation weight
    chi_0_ST = 3.409      # Foreign labor disutility parameter (same as domestic chi_0)
    rho_w_st = 0.9        # Foreign wage persistence (AR(1) approximation)

    # PHASE 2 (Two-Country): Foreign price Calvo parameters (Dynare Eqs. 55-68)
    xi_p_ST = 0.85        # Foreign Calvo price stickiness (same as domestic)
    iota_ST = 0.0         # Foreign price indexation to past inflation (0 = none)
    psi_ST = 0.0          # Foreign Kimball curvature (0 = Dixit-Stiglitz)
    # NKPC slope: kappa = (1-xi)(1-beta*xi)/xi
    kappa_ST = (1 - xi_p_ST) * (1 - beta_ST * xi_p_ST) / xi_p_ST

    # PHASE 5a: Foreign productivity parameters (Dynare Eq. 154)
    rho_z_ST = 0.36637    # Foreign productivity persistence (same as domestic)
    sigma_z_ST = 0.0010004 # Foreign productivity shock std (same as domestic)
    SS_Z_ST = 1.0         # SS foreign productivity

    # PHASE 7: Foreign monetary policy parameters (Taylor rule)
    psi_i_ST = 0.0        # Foreign interest rate smoothing (0 = no smoothing for now)
    psi_pid_ST = 1.5      # Foreign inflation response (same as domestic)
    rho_e_i_ST = 0.0      # Foreign monetary shock persistence
    SS_E_I_ST = 0.0       # SS foreign monetary policy shock

    # Debt limit parameters (OBC)
    m_by = 0.1185         # Distance from debt limit (% of quarterly GDP) - TO BE CALIBRATED
    penalty_kappa =  1.0 #0.1   # Penalty parameter for max(0, -kappa*BLIM) - balance between SS and SEP
    # Note: No SS_Y scaling to avoid circular dependency
    # Note: m and SS_BLIM computed below from SS_NFA and SS_Y

    # *** CONTINUATION PARAMETER ***
    lambda_uip = 1.0      # 0.0 = Nominal UIP, 1.0 = Real UIP

    # *** SEP SHOCK STANDARD DEVIATIONS ***
    # These are used by the SEP solver for Gauss-Hermite quadrature
    # Should match the sigma_ parameters used in shock process equations
    # Note: Use 1e-10 for unused shocks to avoid singular covariance matrix
    z_EPS_B_M = 0.01        # FXI shock (Phase 2)
    z_EPS_B_P = 0.01        # Portfolio shock (Phase 2)
    z_EPS_G = 1e-10         # Government spending shock (not used)
    z_EPS_I = 1e-10         # Monetary policy shock (not used)
    z_EPS_I_ST = 0.001      # Foreign interest rate shock
    z_EPS_NU = 1e-10        # Investment adjustment cost shock (not used)
    z_EPS_PI_ST = 0.001     # Foreign inflation shock
    z_EPS_TAU_C = 1e-10     # Consumption tax shock (not used)
    z_EPS_TAU_F = 0.001     # FX intervention/risk premium shock
    z_EPS_TAU_N = 1e-10     # Labor tax shock (not used)
    z_EPS_UPSILON = 1e-10   # Domestic price markup shock (not used)
    z_EPS_UPSILON_M = 1e-10  # Import price markup shock (very small)
    z_EPS_UPSILON_W = 1e-10  # Wage markup shock (very small)
    z_EPS_VARSIGMA = 0.02971    # Preference shock
    z_EPS_VARSIGMA_ST = 0.027621 # Foreign preference shock (Phase 4)
    z_EPS_Y_ST = 0.01       # Foreign output shock
    z_EPS_Z = 0.0010004     # Productivity shock
    z_EPS_Z_ST = 0.0010004  # Foreign productivity shock (Phase 5a)

    # Steady state values
    SS_E_I = 0.0
    SS_NU = 0.0
    SS_Q = 1.0
    SS_TAU_C = 0.0
    SS_TAU_N = 0.0
    SS_UPSILON = 1 + tau_p
    SS_UPSILON_W = 1 + tau_w
    SS_UPSILON_M = 1 + tau_m
    SS_VARSIGMA = 1.0
    SS_VARTHETA = 1.0
    SS_VARTHETA_M = 1.0
    SS_W_AMP_U = 1.0
    SS_Z = 1.0
    SS_Z_4 = 1.0
    SS_Z_5 = 1.0
    SS_Z_6 = 1.0
    SS_Z_M_5 = 1.0
    SS_Z_M_6 = 1.0
    SS_PI_C_ST = 1.01
    SS_PI_D_ST = 1.01       # Foreign PPI inflation SS (same as CPI in symmetric SS)
    SS_PI_M = 1.01
    SS_GAM_MD = 1.0

    # Derived steady state
    k_n = ((1 + theta_p) / (1 + tau_p) * (1 - beta) / (beta * alpha))^(1 / (alpha - 1))
    SS_MC_D = (1 + tau_p) / (1 + theta_p)
    c_y = 1 - s_gy

    SS_N = (1 / ((c_y + s_gy * eta_0) * (1 - varkappa - SS_NU)) *
           ((1 + tau_p) * (1 + tau_w) * (1 - alpha) / (1 + theta_p) * 1 / chi_0 /
           (1 + theta_w) * (1 - SS_TAU_N) / (1 + SS_TAU_C))^sigma *
           k_n^(alpha * (sigma - 1)))^(1 / (1 + sigma * chi))

    k = k_n * SS_N
    SS_Y = SS_N * k_n^alpha
    SS_C = c_y * SS_Y

    # PHASE 3 (Two-Country): Foreign production parameters (Dynare Eq. 60)
    # Capital-labor ratio (same formula as home but with beta_ST)
    k_n_ST = ((1 + theta_p) / (1 + tau_p) * (1 - beta_ST) / (beta_ST * alpha))^(1 / (alpha - 1))
    # Foreign labor SS (simplified: scale by relative output)
    # SS_N_ST will be solved from: SS_Y_D_ST + SS_Y_X = k_ST^alpha * (SS_Z_ST * SS_N_ST)^(1-alpha)
    # For now, use size_ratio approximation
    SS_N_ST = size_ratio * SS_N
    k_ST = k_n_ST * SS_N_ST
    SS_I = SS_PI_C / beta
    SS_I_ST = SS_PI_C_ST / beta
    SS_IB = SS_I
    SS_PI_D = SS_PI_C
    SS_W_C = k_n^alpha * (1 + tau_p) / (1 + theta_p) * (1 - alpha)
    SS_G = s_gy * SS_Y
    SS_C_TIL = SS_C + eta_0 * SS_G
    SS_LAM = (SS_C_TIL - varkappa * SS_C_TIL - SS_NU * SS_C_TIL)^((-1) / sigma) / (1 + SS_TAU_C)
    SS_W_TIL_C = SS_W_C
    SS_PI_W = SS_PI_C

    z7_corr = chi_0 * SS_W_C^((1 + theta_w) / theta_w * (1 + chi)) * SS_N^(1 + chi)
    z8_corr = SS_N * (1 - SS_TAU_N) * SS_LAM * SS_W_C^((1 + theta_w) / theta_w)

    SS_Z_7 = SS_N^(1 + chi) * SS_W_C^((1 + theta_w) / theta_w * (1 + chi)) * chi_0 * SS_VARSIGMA /
             (1 - beta * xi_w) / z7_corr
    SS_Z_8 = SS_N * SS_W_C^((1 + theta_w) / theta_w) * (1 + tau_w) * (1 - SS_TAU_N) * SS_LAM *
             SS_VARSIGMA / (1 - beta * xi_w) / z8_corr

    # PHASE 3 (Two-Country): Foreign wage and marginal cost steady states
    SS_W_C_ST = k_n_ST^alpha * (1 + tau_p) / (1 + theta_p) * (1 - alpha)
    SS_MC_D_ST = (1 + tau_p) / (1 + theta_p)  # Same as domestic MC in symmetric SS
    SS_W_TIL_C_ST = SS_W_C_ST
    SS_PI_W_ST = SS_PI_C_ST
    SS_W_AMP_U_ST = 1.0

    z7_corr_ST = chi_0_ST * SS_W_C_ST^((1 + theta_w) / theta_w * (1 + chi)) * SS_N_ST^(1 + chi)
    z8_corr_ST = SS_N_ST * (1 - SS_TAU_N_ST) * SS_LAM_ST * SS_W_C_ST^((1 + theta_w) / theta_w)

    SS_Z_7_ST = SS_N_ST^(1 + chi) * SS_W_C_ST^((1 + theta_w) / theta_w * (1 + chi)) * chi_0_ST * SS_VARSIGMA_ST /
                (1 - beta_ST * xi_w_ST) / z7_corr_ST
    SS_Z_8_ST = SS_N_ST * SS_W_C_ST^((1 + theta_w) / theta_w) * (1 + tau_w) * (1 - SS_TAU_N_ST) * SS_LAM_ST *
                SS_VARSIGMA_ST / (1 - beta_ST * xi_w_ST) / z8_corr_ST

    # PHASE 2 (Two-Country): Foreign price Calvo steady states (Dynare Eqs. 55-68)
    SS_P_TIL_D_ST = 1.0       # Foreign optimal reset price ratio
    SS_VARTHETA_ST = 1.0      # Foreign zero profit condition auxiliary
    SS_P_AMP_D_ST = 1.0       # Foreign price dispersion (= 1 in SS)
    SS_Z_4_ST = 1.0           # Foreign price dispersion auxiliary
    SS_Z_5_ST = 1.0           # Foreign price auxiliary
    SS_Z_6_ST = 1.0           # Foreign price auxiliary
    SS_PI_P_ST = SS_PI_D_ST   # Foreign price indexation = PPI in SS

    # Foreign price Calvo auxiliary variables (similar to home Z_1, Z_2, Z_3)
    SS_Z_1_ST = SS_LAM_ST * (1 + theta_p) * (1 + psi_ST) / ((1 - beta_ST * xi_p_ST) * (1 + psi_ST + theta_p * psi_ST)) *
                SS_Y_D_ST * SS_MC_D_ST
    SS_Z_2_ST = SS_Y_D_ST * (1 + tau_p) * SS_LAM_ST / (1 - beta_ST * xi_p_ST)
    SS_Z_3_ST = SS_Y_D_ST * (1 + tau_p) * SS_LAM_ST * theta_p * psi_ST / (1 + psi_ST + theta_p * psi_ST) /
                (1 - beta_ST * xi_p_ST)

    SS_PI_P = SS_PI_D
    SS_PI_PM = SS_PI_M
    SS_U = (log(SS_C_TIL - varkappa * SS_C_TIL - SS_NU * SS_C_TIL) -
           SS_N^(1 + chi) * chi_0 * SS_W_AMP_U / (1 + chi)) / (1 - beta)

    SS_Z_1 = SS_LAM * (1 + theta_p) * (1 + psi) / ((1 - beta * xi_p) * (1 + psi + theta_p * psi)) *
             SS_Y * SS_MC_D
    SS_Z_2 = SS_Y * (1 + tau_p) * SS_LAM / (1 - beta * xi_p)
    SS_Z_3 = SS_Y * (1 + tau_p) * SS_LAM * theta_p * psi / (1 + psi + theta_p * psi) /
             (1 - beta * xi_p)

    # CES steady states (Home)
    SS_GAM_CD = (1 - omega_c + omega_c * SS_GAM_MD^((-1) / rho_c))^(-rho_c)
    SS_GAM_GD = (1 - omega_g + omega_g * SS_GAM_MD^((-1) / rho_g))^(-rho_g)
    SS_GAM_CM = (omega_c + (1 - omega_c) * SS_GAM_MD^(1 / rho_c))^(-rho_c)
    SS_GAM_GM = (omega_g + (1 - omega_g) * SS_GAM_MD^(1 / rho_g))^(-rho_g)

    # PHASE 3 (Two-Country): Foreign CES steady states (Dynare Eqs. 38, 39, 80, 81)
    # Note: In symmetric SS, GAM_MD_ST = 1.0, so these all equal 1.0
    SS_GAM_CD_ST = ((1 - omega_c_ST) + omega_c_ST * SS_GAM_MD_ST^((-1) / rho_c_ST))^(-rho_c_ST)
    SS_GAM_GD_ST = ((1 - omega_g_ST) + omega_g_ST * SS_GAM_MD_ST^((-1) / rho_g_ST))^(-rho_g_ST)
    SS_GAM_CM_ST = ((1 - omega_c_ST) * SS_GAM_MD_ST^(1 / rho_c_ST) + omega_c_ST)^(-rho_c_ST)
    SS_GAM_GM_ST = ((1 - omega_g_ST) * SS_GAM_MD_ST^(1 / rho_g_ST) + omega_g_ST)^(-rho_g_ST)

    SS_M_C = SS_C * omega_c * SS_GAM_CM^((1 + rho_c) / rho_c)
    SS_M_G = SS_G * omega_g * SS_GAM_GM^((1 + rho_g) / rho_g)
    SS_Y_M_ST = SS_M_C + SS_M_G

    SS_Y_D = SS_C * (1 - omega_c) * SS_GAM_CD^((1 + rho_c) / rho_c) +
             SS_G * (1 - omega_g) * SS_GAM_GD^((1 + rho_g) / rho_g)

    SS_P_TIL_M = 1.0
    SS_Z_M_1 = SS_Q * SS_LAM * (1 + theta_p) * (1 + psi_m) / ((1 - beta * xi_m) * (1 + psi_m + theta_p * psi_m)) *
               SS_Y_M_ST / SS_GAM_CD
    SS_Z_M_2 = SS_Y_M_ST * SS_LAM * (1 + tau_m) * SS_Q / (1 - beta * xi_m) / SS_GAM_CM
    SS_Z_M_3 = SS_Q * SS_Y_M_ST * (1 + tau_m) * SS_LAM * theta_p * psi_m / (1 + psi_m + theta_p * psi_m) /
               (1 - beta * xi_m) / SS_GAM_CM

    SS_Y_X = SS_Y - SS_Y_D
    SS_TB = SS_Y_X - SS_Q * SS_Y_M_ST
    SS_NFA = 0.0

    # Debt limit parameters (derived from SS_NFA and SS_Y)
    # m_by = 0.1185 means debt limit is ~47% of annual GDP above steady state
    # (multiply by 4 to convert quarterly to annual)
    m = -SS_NFA/SS_Y + m_by*4  # Debt limit distance parameter
    SS_BLIM = SS_NFA + m * SS_Y # Steady state debt limit

    # ========================================================================
    # PHASE 6: Structural Foreign Sector Steady States
    # ========================================================================
    # In Dynare, SS_C_ST is pinned by trade balance. Here we derive it from:
    # Y_ST = Y_D_ST + Y_M_ST
    # Y_D_ST = (1 - omega_c_ST) * C_ST
    # Therefore: C_ST = (Y_ST - Y_M_ST) / (1 - omega_c_ST)
    #
    # We set SS_Y_ST exogenously (small open economy assumption)
    SS_Y_ST = size_ratio * SS_Y

    # SS_C_ST derived from structural relationship
    # Y_ST = Y_D_ST + Y_M_ST = (1 - omega_c_ST) * C_ST + Y_M_ST
    # C_ST = (Y_ST - Y_M_ST) / (1 - omega_c_ST)
    SS_C_ST = (SS_Y_ST - SS_Y_M_ST) / (1 - omega_c_ST)

    # Foreign domestic demand (from Y_D_ST equation, using GAM_CD_ST)
    # Note: SS_GAM_CD_ST^((1+rho_c_ST)/rho_c_ST) = 1.0 when SS_GAM_MD_ST = 1.0
    SS_Y_D_ST = (1 - omega_c_ST) * SS_GAM_CD_ST^((1 + rho_c_ST) / rho_c_ST) * SS_C_ST

    # Foreign household steady states (needed for wage Calvo)
    SS_G_ST = 0.0  # Foreign government spending (set to 0 for small open economy)
    SS_C_TIL_ST = SS_C_ST + eta_0 * SS_G_ST
    SS_NU_ST = 0.0  # Foreign preference shock SS
    SS_LAM_ST = (SS_C_TIL_ST - varkappa_ST * SS_C_TIL_ST - SS_NU_ST * SS_C_TIL_ST)^((-1) / sigma) / (1 + TAU_C_ST)

    # Variable bounds to help SEP solver with OBC
    THETA >= 0.0  # Risk premium must be non-negative
end
