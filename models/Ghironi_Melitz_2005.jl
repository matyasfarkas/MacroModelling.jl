@model Ghironi_Melitz_2005 begin
	1 = Nd[0] * ρ̃d[0] ^ (1 - θ) + Nx̄[0] * ρ̃x̄[0] ^ (1 - θ)

	1 = Nd̄[0] * ρ̃d̄[0] ^ (1 - θ) + Nx[0] * ρ̃x[0] ^ (1 - θ)

	ρ̃d[0] = θ / (θ - 1) * w[0] / (Z[0] * z̃d)

	ρ̃d̄[0] = θ / (θ - 1) * w̄[0] / (Z̄[0] * z̃d̄)

	ρ̃x[0] = θ / (θ - 1) * τ * w[0] / (Z[0] * z̃x[0]) / Q[0]

	ρ̃x̄[0] = Q[0] * θ / (θ - 1) * τ * w̄[0] / (Z̄[0] * z̃x̄[0])

	d̃[0] = d̃d[0] + Nx[0] / Nd[0] * d̃x[0]

	d̃̄[0] = d̃d̄[0] + Nx̄[0] / Nd̄[0] * d̃x̄[0]

	d̃d[0] = ρ̃d[0] ^ (1 - θ) * 1 / θ * C[0]

	d̃d̄[0] = ρ̃d̄[0] ^ (1 - θ) * 1 / θ * C̄[0]

	ṽ[0] = w[0] * fe / Z[0]

	ṽ̄[0] = w̄[0] * fē / Z̄[0]

	d̃x[0] = w[0] * fx / Z[0] * (θ - 1) / (k - (θ - 1))

	d̃x̄[0] = (θ - 1) / (k - (θ - 1)) * w̄[0] * fx̄ / Z̄[0]

	Nx[0] / Nd[0] = (zmin / z̃x[0]) ^ k * (k / (k - (θ - 1))) ^ (k / (θ - 1))

	Nx̄[0] / Nd̄[0] = (k / (k - (θ - 1))) ^ (k / (θ - 1)) * (zmin̄ / z̃x̄[0]) ^ k

	Nd[0] = (1 - δ) * (Nd[-1] + Ne[-1])

	Nd̄[0] = (1 - δ) * (Nd̄[-1] + Nē[-1])

	C[0] ^ (-γ) = β * (1 + r[0]) * C[1] ^ (-γ)

	C̄[0] ^ (-γ) = β * (1 + r̄[0]) * C̄[1] ^ (-γ)

	ṽ[0] = (1 - δ) * β * (C[1] / C[0]) ^ (-γ) * (ṽ[1] + d̃[1])

	ṽ̄[0] = (1 - δ) * β * (C̄[1] / C̄[0]) ^ (-γ) * (ṽ̄[1] + d̃̄[1])

	C[0] = w[0] * L + Nd[0] * d̃[0] - ṽ[0] * Ne[0]

	C̄[0] = w̄[0] * L̄ + Nd̄[0] * d̃̄[0] - ṽ̄[0] * Nē[0]

	Q[0] = Nx̄[0] * ρ̃x̄[0] ^ (1 - θ) * C[0] / (Nx[0] * ρ̃x[0] ^ (1 - θ) * C̄[0])

	Q̃[0] = ((Nd̄[0] / (Nd̄[0] + Nx[0]) * TOL[0] ^ (1 - θ) + Nx[0] / (Nd̄[0] + Nx[0]) * (τ * z̃d / z̃x[0]) ^ (1 - θ)) / (Nd[0] / (Nd[0] + Nx̄[0]) + Nx̄[0] / (Nd[0] + Nx̄[0]) * (τ * TOL[0] * z̃d̄ / z̃x̄[0]) ^ (1 - θ))) ^ (1 / (1 - θ))

	Q̃[0] = Q[0] * ((Nd[0] + Nx̄[0]) / (Nd̄[0] + Nx[0])) ^ (( - 1) / (θ - 1))

	Z[0] = (1 - ρZ) * 1.0 + ρZ * Z[-1] + σᶻ * ϵᶻ[x]

	Z̄[0] = 1.0 * (1 - ρZ̄) + ρZ̄ * Z̄[-1] + σᶻ̄ * ϵᶻ̄[x]

	z̃x[0] = (θ * fx * (w[0] / Z[0]) ^ θ * (1 + (θ - 1) / (k - (θ - 1))) * Q[0] ^ (-θ) * τ ^ (θ - 1) * (θ / (θ - 1)) ^ (θ - 1) * C̄[0] ^ (-1)) ^ (1 / (θ - 1))

	z̃x̄[0] = ((θ / (θ - 1)) ^ (θ - 1) * θ * τ ^ (θ - 1) * (1 + (θ - 1) / (k - (θ - 1))) * fx̄ * (w̄[0] / Z̄[0]) ^ θ * Q[0] ^ θ * C[0] ^ (-1)) ^ (1 / (θ - 1))

	zx[0] = z̃x[0] / (k / (k - (θ - 1))) ^ (1 / (θ - 1))

	zx̄[0] = z̃x̄[0] / (k / (k - (θ - 1))) ^ (1 / (θ - 1))

end


@parameters Ghironi_Melitz_2005 begin
    σᶻ = .01

    σᶻ̄ = .01

    fx = fx_share * (1 - β * (1 - δ)) / (β * (1 - δ)) * fe

    fx̄ = fx_share * (1 - β * (1 - δ)) / (β * (1 - δ)) * fē

	β = 0.99

	γ = 2.0

	δ = 0.025

	θ = 3.8

	k = 3.4

	τ = 1.3

	zmin = 1.0

	zmin̄ = 1.0

	z̃d = (k / (k - (θ - 1))) ^ (1 / (θ - 1)) * zmin

	z̃d̄ = (k / (k - (θ - 1))) ^ (1 / (θ - 1)) * zmin̄

	fe = 1.0

	fē = 1.0

	L = 1.0

	L̄ = 1.0

	ρZ = 0.9

	ρZ̄ = 0.9

	fx_share = 0.235

end

