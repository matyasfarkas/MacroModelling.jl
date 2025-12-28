using CSV
using DataFrames
using Printf

# Load CSV
benchmark_path = "/Users/matyasfarkas/Documents/GitHub/Non_Linear_DSGE/SEP/RBC_irf.csv"
df = CSV.read(benchmark_path, DataFrame, header=3, skipto=3, types=Float64, silencewarnings=true)

# Extract Output data (column 2 = ts, column 11 = tt)
ts_output = df[:, 2]  # All rows, column 2
tt_output_full = df[:, 11]  # All rows, column 11

# Remove missing values from tt (it only has 81 periods)
tt_output = collect(skipmissing(tt_output_full))

# Apply pdss transformation: 100*(data - data[1])/data[1]
function pdss(data)
    return 100 .* (data .- data[1]) ./ data[1]
end

pdss_ts = pdss(ts_output)
pdss_tt = pdss(tt_output)

# Compute difference
irf_diff = pdss_tt - pdss_ts[1:length(pdss_tt)]

println("="^80)
println("COMPUTED IRF VALUES FROM CSV")
println("="^80)

println("\nFirst 20 periods:")
println("\nPeriod  |  tt.Output  |  ts.Output  |  pdss(tt)  |  pdss(ts)  |  Difference")
println("-"^80)

for i in 1:min(20, length(pdss_tt))
    @printf("%6d  |  %10.6f  |  %10.6f  |  %9.4f  |  %9.4f  |  %9.6f\n", 
            i-1, tt_output[i], ts_output[i], pdss_tt[i], pdss_ts[i], irf_diff[i])
end

println("\n" * "="^80)
println("COMPARISON WITH YOUR REPORTED VALUES:")
println("="^80)
user_irf = [0, 63.4581, 50.0188, 39.6554, 31.6213, 25.3604, 20.4574, 16.6010, 13.5558, 11.1426, 9.2240, 7.6941, 6.4706, 5.4892, 4.6997, 4.0626]

println("\nPeriod  |  Your IRF  |  pdss(tt)  |  pdss(ts)  |  -pdss(tt)  |  -pdss(ts)")
println("-"^80)
for i in 1:min(16, length(user_irf))
    @printf("%6d  |  %9.4f  |  %9.4f  |  %9.4f  |  %10.4f  |  %10.4f\n", 
            i-1, user_irf[i], pdss_tt[i], pdss_ts[i], -pdss_tt[i], -pdss_ts[i])
end

# Check efficiency to see which shock this is
efficiency = df[:, 6]  # Column 6
println("\n" * "="^80)
println("CHECKING SHOCK SIGN:")
println("="^80)
println("\nefficiency at period 0: $(efficiency[1])")
println("efficiency at period 1: $(efficiency[2])")
println("This corresponds to a $(efficiency[2] < 0 ? "NEGATIVE" : "POSITIVE") shock")

println("\n" * "="^80)
println("INSIGHT:")
println("="^80)
println("If this is a NEGATIVE shock (-3σ) but you report POSITIVE IRF values,")
println("then maybe you saved the CSV from a DIFFERENT run with POSITIVE shock (+3σ)?")
println("Or the IRF definition involves taking the negative: IRF = -pdss(path)?")
