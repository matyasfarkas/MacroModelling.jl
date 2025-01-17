### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ eae9eb99-17c3-4b49-a533-fde043c80b12
using MAT

# ╔═╡ 3acd62cc-fd47-4f87-a644-bad1a567f098
using SparseArrays

# ╔═╡ 6dfe2cb9-88bb-4667-8951-03f42ec2688f
using PlutoUI

# ╔═╡ f708dc72-0ad7-4457-b6b4-160ffbc3732a
md"# Macroeconomic Database Replication 
## Replicating the functionality of the Macroeconomic Database in MacroModelling.jl
It is super simple, we first need to load in the respecitve packages. First load in MacroModelling.jl.
The TOM at the end of this file should control for the environment and other dependencies to be installed."


# ╔═╡ d868948e-c592-41c4-8dfd-52d58519fc34
import MacroModelling

# ╔═╡ dccf88e3-403f-4af4-9744-1bde70ec0089
md" Load in StatsPolts for the IRF charts."

# ╔═╡ 287b3ff0-913a-4e64-ba5b-c829c307ec70
import StatsPlots

# ╔═╡ 78b0fa66-629e-411c-9a8a-c4cf92957b39
md" Define the SW07 model."

# ╔═╡ 58c973ac-291b-401f-90b1-69eba41c104b
@model SW07 begin
    a[0] = calfa * rkf[0] + (1 - calfa) * (wf[0])

    zcapf[0] = (1 / (czcap / (1 - czcap))) * rkf[0]

    rkf[0] = wf[0] + labf[0] - kf[0]

    kf[0] = kpf[-1] + zcapf[0]

    invef[0] = (1 / (1 + cbetabar * cgamma)) * (invef[-1] + cbetabar * cgamma * invef[1] + (1 / (cgamma ^ 2 * csadjcost)) * pkf[0]) + qs[0]

    pkf[0] =  - rrf[0] + (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) * b[0] + (crk / (crk + (1 - ctou))) * rkf[1] + ((1 - ctou) / (crk + (1 - ctou))) * pkf[1]

    cf[0] = (chabb / cgamma) / (1 + chabb / cgamma) * cf[-1] + (1 / (1 + chabb / cgamma)) * cf[1] + ((csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma))) * (labf[0] - labf[1]) - (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)) * (rrf[0]) + b[0]

    yf[0] = ccy * cf[0] + ciy * invef[0] + g[0] + crkky * zcapf[0]

    yf[0] = cfc * (calfa * kf[0] + (1 - calfa) * labf[0] + a[0])

    wf[0] = csigl * labf[0]	 + (1 / (1 - chabb / cgamma)) * cf[0] - (chabb / cgamma) / (1 - chabb / cgamma) * cf[-1]

    kpf[0] = (1 - cikbar) * kpf[-1] + (cikbar) * invef[0] + (cikbar) * (cgamma ^ 2 * csadjcost) * qs[0]

    mc[0] = calfa * rk[0] + (1 - calfa) * (w[0]) - a[0]

    zcap[0] = (1 / (czcap / (1 - czcap))) * rk[0]

    rk[0] = w[0] + lab[0] - k[0]

    k[0] = kp[-1] + zcap[0]

    inve[0] = (1 / (1 + cbetabar * cgamma)) * (inve[-1] + cbetabar * cgamma * inve[1] + (1 / (cgamma ^ 2 * csadjcost)) * pk[0]) + qs[0]

    pk[0] =  - r[0] + pinf[1] + (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) * b[0] + (crk / (crk + (1 - ctou))) * rk[1] + ((1 - ctou) / (crk + (1 - ctou))) * pk[1]

    c[0] = (chabb / cgamma) / (1 + chabb / cgamma) * c[-1] + (1 / (1 + chabb / cgamma)) * c[1] + ((csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma))) * (lab[0] - lab[1]) - (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)) * (r[0] - pinf[1]) + b[0]

    y[0] = ccy * c[0] + ciy * inve[0] + g[0] + crkky * zcap[0]

    y[0] = cfc * (calfa * k[0] + (1 - calfa) * lab[0] + a[0])

    pinf[0] = (1 / (1 + cbetabar * cgamma * cindp)) * (cbetabar * cgamma * pinf[1] + cindp * pinf[-1] + ((1 - cprobp) * (1 - cbetabar * cgamma * cprobp) / cprobp) / ((cfc - 1) * curvp + 1) * (mc[0])) + spinf[0]

    w[0] = (1 / (1 + cbetabar * cgamma)) * w[-1] + (cbetabar * cgamma / (1 + cbetabar * cgamma)) * w[1] + (cindw / (1 + cbetabar * cgamma)) * pinf[-1] - (1 + cbetabar * cgamma * cindw) / (1 + cbetabar * cgamma) * pinf[0] + (cbetabar * cgamma) / (1 + cbetabar * cgamma) * pinf[1] + (1 - cprobw) * (1 - cbetabar * cgamma * cprobw) / ((1 + cbetabar * cgamma) * cprobw) * (1 / ((clandaw - 1) * curvw + 1)) * (csigl * lab[0] + (1 / (1 - chabb / cgamma)) * c[0] - ((chabb / cgamma) / (1 - chabb / cgamma)) * c[-1] - w[0]) + sw[0]

    r[0] = crpi * (1 - crr) * pinf[0] + cry * (1 - crr) * (y[0] - yf[0]) + crdy * (y[0] - yf[0] - y[-1] + yf[-1]) + crr * r[-1] + ms[0]

    a[0] = crhoa * a[-1] + z_ea * ea[x]

    b[0] = crhob * b[-1] + z_eb * eb[x]

    g[0] = crhog * g[-1] + z_eg * eg[x] + cgy * z_ea * ea[x]

    qs[0] = crhoqs * qs[-1] + z_eqs * eqs[x]

    ms[0] = crhoms * ms[-1] + z_em * em[x]

    spinf[0] = crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]

    epinfma[0] = z_epinf * epinf[x]

    sw[0] = crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]

    ewma[0] = z_ew * ew[x]

    kp[0] = (1 - cikbar) * kp[-1] + cikbar * inve[0] + cikbar * cgamma ^ 2 * csadjcost * qs[0]

    dy[0] = y[0] - y[-1] + ctrend

    dc[0] = c[0] - c[-1] + ctrend

    dinve[0] = inve[0] - inve[-1] + ctrend

    dw[0] = w[0] - w[-1] + ctrend

    pinfobs[0] = (pinf[0]) + constepinf

    robs[0] = (r[0]) + conster

    labobs[0] = lab[0] + constelab

end


# ╔═╡ 3a49e273-21e2-4a20-a641-d3545ef6f485
@bind crr Slider(0.1:0.01:0.99)

# ╔═╡ 9efd561e-af30-490d-9881-8916190c34dc
@bind csigma Slider(0.1:0.01:12)


# ╔═╡ c2ef96aa-d73d-412f-8ff4-96a073dbf0b4
plot_irf(SW07, parameters = [:crr => crr, :crpi => crpi, :cry => cry, :csigma => csigma],periods = 40)[1]


# ╔═╡ 65d6da83-9474-4ad5-9deb-b1e35088fa51
get_SS(SW07)

md"Use slides for the parameters.
Starting with crpi - the coefficient on inflation deviation from target in the TR:"


# ╔═╡ c284e40c-a57b-42e2-b62f-24791510c9cd
@bind cry Slider(0.:0.01:0.1)

md"Finally with crr - the coefficient on interest rate smoothing in the TR:"

# ╔═╡ b4f4ce24-5162-41b9-a36c-09af8f7b041b
@parameters SW07 begin  
    ctou=.025
    clandaw=1.5
    cg=0.18
    curvp=10
    curvw=10
    
    calfa=.24
    csigma=1.5
    cfc=1.5
    cgy=0.51
    
    csadjcost= 6.0144
    chabb=    0.6361    
    cprobw=   0.8087
    csigl=    1.9423
    cprobp=   0.6
    cindw=    0.3243
    cindp=    0.47
    czcap=    0.2696
    crpi=     1.488
    crr=      0.8762
    cry=      0.0593
    crdy=     0.2347
    
    crhoa=    0.9977
    crhob=    0.5799
    crhog=    0.9957
    crhols=   0.9928
    crhoqs=   0.7165
    crhoas=1 
    crhoms=0
    crhopinf=0
    crhow=0
    cmap = 0
    cmaw  = 0
    
    clandap=cfc
    cbetabar=cbeta*cgamma^(-csigma)
    cr=cpie/(cbeta*cgamma^(-csigma))
    crk=(cbeta^(-1))*(cgamma^csigma) - (1-ctou)
    cw = (calfa^calfa*(1-calfa)^(1-calfa)/(clandap*crk^calfa))^(1/(1-calfa))
    cikbar=(1-(1-ctou)/cgamma)
    cik=(1-(1-ctou)/cgamma)*cgamma
    clk=((1-calfa)/calfa)*(crk/cw)
    cky=cfc*(clk)^(calfa-1)
    ciy=cik*cky
    ccy=1-cg-cik*cky
    crkky=crk*cky
    cwhlc=(1/clandaw)*(1-calfa)/calfa*crk*cky/ccy
    cwly=1-crk*cky
    
    conster=(cr-1)*100
    ctrend=(1.004-1)*100
    constepinf=(1.005-1)*100

    cpie=1+constepinf/100
    cgamma=1+ctrend/100 

    cbeta=1/(1+constebeta/100)
    constebeta = 100 / .9995 - 100

    constelab=0

    z_ea = 0.4618
    z_eb = 1.8513
    z_eg = 0.6090
    z_eqs = 0.6017
    z_em = 0.2397
    z_epinf = 0.1455
    z_ew = 0.2089
end

md"Solve the SW07 model."

# ╔═╡ ee54cb85-b7fe-4638-919b-9d14779c7502
@bind crpi Slider(1.1:0.01:1.9)

md"Following with cry - the coefficient on output gap in the TR:"

# ╔═╡ Cell order:
# ╟─f708dc72-0ad7-4457-b6b4-160ffbc3732a
# ╠═d868948e-c592-41c4-8dfd-52d58519fc34
# ╠═dccf88e3-403f-4af4-9744-1bde70ec0089
# ╠═287b3ff0-913a-4e64-ba5b-c829c307ec70
# ╠═78b0fa66-629e-411c-9a8a-c4cf92957b39
# ╠═eae9eb99-17c3-4b49-a533-fde043c80b12
# ╠═3acd62cc-fd47-4f87-a644-bad1a567f098
# ╠═6dfe2cb9-88bb-4667-8951-03f42ec2688f
# ╠═58c973ac-291b-401f-90b1-69eba41c104b
# ╠═3a49e273-21e2-4a20-a641-d3545ef6f485
# ╠═9efd561e-af30-490d-9881-8916190c34dc
# ╠═c2ef96aa-d73d-412f-8ff4-96a073dbf0b4
# ╠═65d6da83-9474-4ad5-9deb-b1e35088fa51
# ╠═c284e40c-a57b-42e2-b62f-24791510c9cd
# ╠═b4f4ce24-5162-41b9-a36c-09af8f7b041b
# ╠═ee54cb85-b7fe-4638-919b-9d14779c7502