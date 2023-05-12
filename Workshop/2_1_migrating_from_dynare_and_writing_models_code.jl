#= 
# Policy applications I
## Migrating an existing model to MacroModelling.jl from dynare

Let's try to translate the basic new keynesian textbook model from Gali Chapter 3 in it's nonlinear version. We can get the mod file [here](http://www.thorekockerols.eu/models/Gali_2015_chapter_3_nonlinear.mod)

Once we downloaded the mod file we start by loading the package:"
=#

using MacroModelling
using SparseArrays

# Print the current working directory:
pwd()

# Change the current working directory:
cd()


name_of_mod_file_including_path = "Gali_2015_chapter_3_nonlinear.mod"

# The package includes a function to translate model equations and parameters from a dynare mod file into the format for MacroModelling.jl:
import_model(name_of_mod_file_including_path)


# "Now that we translated the mod file let's compare the two:"
run(`open $name_of_mod_file_including_path`)


name_of_jl_file_including_path = "Gali_2015_chapter_3_nonlinear.jl"

run(`open $name_of_jl_file_including_path`)


# Let's try to run the model then:
include(name_of_jl_file_including_path)


# It does not work out of the box. Now have a ook at the error message and see if what is written there applies to the model at hand.



# You can make changes to the model either in the newly created file or here:
@model Gali_2015_chapter_3_nonlinear_fix begin
	W_real[0] = C[0] ^ siggma * N[0] ^ varphi

	Q[0] = betta * (C[1] / C[0]) ^ (-siggma) * Z[1] / Z[0] / Pi[1]

	R[0] = 1 / Q[0]

	Y[0] = A[0] * (N[0] / S[0]) ^ (1 - alppha)

	R[0] = Pi[1] * realinterest[0]

	R[0] = 1 / betta * Pi[0] ^ phi_pi * (Y[0] / Y[ss]) ^ phi_y * exp(nu[0])

	C[0] = Y[0]

	log(A[0]) = rho_a * log(A[-1]) + eps_a[x]

	log(Z[0]) = rho_z * log(Z[-1]) - eps_z[x]

	nu[0] = rho_nu * nu[-1] + eps_nu[x]

	MC[0] = W_real[0] / (S[0] * Y[0] * (1 - alppha) / N[0])

	1 = theta * Pi[0] ^ (epsilon - 1) + (1 - theta) * Pi_star[0] ^ (1 - epsilon)

	S[0] = (1 - theta) * Pi_star[0] ^ (( - epsilon) / (1 - alppha)) + theta * Pi[0] ^ (epsilon / (1 - alppha)) * S[-1]

	Pi_star[0] ^ (1 + epsilon * alppha / (1 - alppha)) = epsilon * x_aux_1[0] / x_aux_2[0] * (1 - tau) / (epsilon - 1)

	x_aux_1[0] = MC[0] * Y[0] * Z[0] * C[0] ^ (-siggma) + betta * theta * Pi[1] ^ (epsilon + alppha * epsilon / (1 - alppha)) * x_aux_1[1]

	x_aux_2[0] = Y[0] * Z[0] * C[0] ^ (-siggma) + betta * theta * Pi[1] ^ (epsilon - 1) * x_aux_2[1]

	log_y[0] = log(Y[0])

	log_W_real[0] = log(W_real[0])

	log_N[0] = log(N[0])

	pi_ann[0] = 4 * log(Pi[0])

	i_ann[0] = 4 * log(R[0])

	r_real_ann[0] = 4 * log(realinterest[0])

	M_real[0] = Y[0] / R[0] ^ eta

	log_m_nominal[0] = log(M_real[0] * P[0])

	Pi[0] = P[0] / P[-1]

	log_P[0] = log(P[0])

	log_A[0] = log(A[0])

	log_Z[0] = log(Z[0])

end



@parameters Gali_2015_chapter_3_nonlinear_fix begin
	siggma = 1

	varphi = 5

	phi_pi = 1.5

	phi_y = 0.125

	theta = 0.75

	rho_nu = 0.5

	rho_z = 0.5

	rho_a = 0.9

	betta = 0.99

	eta = 3.77

	alppha = 0.25

	epsilon = 9

	tau = 0

end


# Once the necessary modifications are done you can go on and get the steady state. Check the documentation for the relevant function: http://www.thorekockerols.eu/MacroModelling.jl/stable/call_index/



# and plot IRFs. In order to do plots you need to load the StatsPlots package first:
import StatsPlots

# Try to play around with the options of the irf plot command. Check the documentation in the console with: ?<name of command for which you want to see the documentation>


# You can export the model as a mod file as well:
write_mod_file(Gali_2015_chapter_3_nonlinear_fix)

# which looks like this then:
path_to_mod_file = "Gali_2015_chapter_3_nonlinear_fix.mod"

run(`open $path_to_mod_file`)




## Some Julia basics
# Creating vectors and arrays


V = [0, 1, 2]

M = [1 2 3
	 3 4 5]

M̂ = [1 2 3 ;3 4 5]

V2 = rand(3)

M2 = rand(3,3)

Vsparse = spzeros(3)

Msparse = spzeros(3,3)

Vkeyed = KeyedArray(rand(2), Variable = [:K, :Y])

Mkeyed = KeyedArray(rand(2,2), Variable = [:K, :Y], Shock = [:ϵ, :ε])


# Accessing and modifying vectors and matrices


V[1]

V[1:2]

V[2] = -9

M[1,2]

M[2,2:3]

M[2,2:3] .= 14

# M[2,2:3] = 14 doesnt work

Msparse[1,2] = 10

Msparse

findnz(Msparse)

Vkeyed(:K)

Vkeyed[1]

Vkeyed[1] = -12


# Vkeyed(:K) = 1 # doesnt work

axiskeys(Vkeyed)

Mkeyed(:K,:ϵ)

Mkeyed[1,2]

axiskeys(Mkeyed)


#= 
## Writing your own first model
Let's say you have the following simple real businesss cycle model you want to implement:

$$\begin{align}
K_{t}&=\left(1-\delta\right)K_{t-1} + I_{t}\\
Y_{t} &= Z_{t}  K_{t-1}^\alpha\\
Y_{t} &= C_{t} + I_{t}\\
\frac{1}{C_{t}^\gamma}&=\frac{\beta}{C_{t+1}^\gamma}  \left(\alpha  \frac{Y_{t+1}}{K_{t}} + \left( 1 - \delta \right)\right)\\
Z_{t} &= \left( 1 - \rho \right) + \rho  Z_{t-1} + \sigma \epsilon_{t}
\end{align}$$

with parameter values:

$$\begin{align}
\alpha &= \frac{1}{3}\\
\beta &= 0.99\\
\delta &= 0.025\\
\gamma &= 1
\\
\rho &= 0.7\\
\sigma &= 0.005
=#

# Check the help (?) for @model and @parameters


# Once you wrote the model try to print the non stochastic steady state, plot IRFs, and the the theoretical model covariance



# Next lets try to calibrate the steady state such that the capital to yearly GDP ratio is 1.66
# Copy the previously written model (both @model and @parameters parts) and do the necessary changes in the @parameters part. Check the help (?) for @parameters

# Once you wrote the model try to print the non stochastic steady state, plot IRFs, and the the theoretical model covariance