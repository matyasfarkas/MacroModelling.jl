# Dynare RS Model - Recursive Utility DSGE
# Translated from rs.mod for Dynare SEP validation

using MacroModelling

@model DynareRS begin
    # Value function and Euler equation
    V[0] = exp(lC[0])^(1-phi)/(1-phi) + chi0*(LMax-exp(lL[0]))^(1-chi)/(1-chi) + beta*Vkp[0]

    Int1[0] = Int[0]/10
    lpi1[0] = lpi[0]/10

    exp(lC[0])^(-phi) = beta*exp(10*Int1[0]-10*lpi1[1])*exp(lC[1])^(-phi)*exp(lDZ[1])^(-phi)*(V[1]*exp(lDZ[1])^(1-phi)/Vkp[0])^(-alpha)

    # E-Z-W-K-P certainty equivalent
    Valphaexp[0] = (V[1]*exp(lDZ[1])^(1-phi)/VAIMSS/DZBar^(1-phi))^(1-alpha)
    Vkp[0] = VAIMSS*DZBar^(1-phi)*Valphaexp[0]^(1/(1-alpha))

    # Price setting
    exp(lzn[0]) = (1+theta)*MC[0]*exp(lY[0]) + xi*beta*exp(lC[1]-lC[0])^(-phi)*exp(lDZ[1])^(-phi)*(V[1]*exp(lDZ[1])^(1-phi)/Vkp[0])^(-alpha)*exp(lpi[1])^((1+theta)/theta/eta)*exp(lzn[1])

    exp(lzd[0]) = exp(lY[0]) + xi*beta*exp(lC[1]-lC[0])^(-phi)*exp(lDZ[1])^(-phi)*(V[1]*exp(lDZ[1])^(1-phi)/Vkp[0])^(-alpha)*exp(lpi[1])^(1/theta)*exp(lzd[1])

    exp(lp0[0])^(1+(1+theta)/theta*(1-eta)/eta) = exp(lzn[0]-lzd[0])

    exp(lpi[0])^(-1/theta) = (1-xi)*exp(lp0[0]+lpi[0])^(-1/theta) + xi

    # Marginal cost and real wage
    MC[0] = exp(lwreal[0])/eta*exp(lY[0])^((1-eta)/eta)/exp(lA[0])^(1/eta)/KBar^((1-eta)/eta)

    chi0*(LMax-exp(lL[0]))^(-chi)/exp(lC[0])^(-phi) = exp(lwreal[0])

    # Output equations
    exp(lY[0]) = exp(lA[0])*KBar^(1-eta)*exp(lL[0])^eta/exp(lDisp[0])

    exp(lDisp[0])^(1/eta) = (1-xi)*exp(lp0[0])^(-(1+theta)/theta/eta) + xi*exp(lpi[0])^((1+theta)/theta/eta)*exp(lDisp[-1])^(1/eta)

    exp(lC[0]) = exp(lY[0]) - exp(lG[0]) - IBar

    # Monetary policy
    lpiavg[0] = rhoinflavg*lpiavg[-1] + (1-rhoinflavg)*lpi[0]

    4*Int[0] = (1-taylrho)*(4*log(1/beta*DZBar^phi) + 4*lpiavg[0] + taylpi*(4*lpiavg[0]-pistar[0]) + tayly*(exp(lY[0])-YBar)/YBar) + taylrho*4*Int[-1] + epsInt[x]

    # Exogenous shocks
    lA[0] = rhoa*lA[-1] + epsA[x]
    lDZ[0] = (1-rhoz)*log(DZBar) + rhoz*lDZ[-1]
    lG[0] = (1-rhog)*log(GBar) + rhog*lG[-1] + epsG[x]
    pistar[0] = (1-rhopistar)*log(piBar) + rhopistar*pistar[-1] + gssload*(4*lpiavg[0]-pistar[0])

    # Auxiliary variables
    Intr[0] = Int[-1] - lpi[0]
    exp(lsdf[0]) = beta*exp(lC[1]-lC[0])^(-phi)*exp(lDZ[1])^(-phi)*(V[1]*exp(lDZ[1])^(1-phi)/Vkp[0])^(-alpha)/exp(lpi[1])
end

@parameters DynareRS begin
    DZBar = 1.0025
    eta = 2/3
    IES = 11/100
    phi = 1/IES
    beta = 99/100*DZBar^phi
    delta = 2/100
    Frisch = 28/100
    CRRA = 110
    LMax = 3
    chi0 = 1/3
    chi = 1/Frisch*(LMax-1)
    alpha = (CRRA-1/(1/phi+(LMax-1)*1/chi))*(1/(1-phi)+(LMax-1)*1/(1-chi))
    theta = 2/10
    xi = 78/100
    K_Y = 10
    YBar = K_Y^((1-eta)/eta)
    GBar = 17/100*YBar
    KBar = K_Y*YBar
    IBar = (delta+(DZBar-1))*KBar
    piBar = 1
    taylrho = 73/100
    taylpi = 53/100
    tayly = 93/100
    rhoa = 96/100
    rhoz = 0
    rhog = 95/100
    rhopistar = 0
    rhoinflavg = 7/10
    gssload = 0
    VAIMSS = 1
    z_epsA = 1/1000
    z_epsG = 4/1000
    z_epsInt = 3/1000
end
