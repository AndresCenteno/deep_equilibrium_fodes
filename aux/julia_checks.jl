include("mittag_leffler.jl")
using LinearAlgebra
using SpecialFunctions: gamma

d = 3
A = randn(d,d);
Λ, Χ = eigen(A)
T = 0.1
u0 = randn(d)
α = 0.8

uT = Χ*diagm(mittleff.(α,T^α.*Λ))*(Χ\u0)

Nt = 400

# first, a forward pass based on the integral formulation
function f(A::Matrix,z::Matrix,x::Vector,T,α)
    Nt = size(z,2)
    Δt = T/Nt
    z_out = zeros(size(z))
    for nt = 1:Nt
        t = nt*Δt
        z_out[:,nt] = x + Δt*A*x*t^(α-1)/gamma(α)
        for j = 1:(nt-1)
            z_out[:,nt] += Δt*A*z_out[:,j]*(t-Δt*j)^(α-1)/gamma(α)
        end 
    end
    return z_out
end

function g(A::Matrix,z::Matrix,x::Vector,T,α)
    Nt = size(z,2)
    Δt = T/Nt
    z_out = zeros(size(z))
    for nt = 1:Nt
        t = nt*Δt
        z_out[:,nt] = x + Δt*A*x*Δt^(α-1)/gamma(α)
        for j = 1:(nt-1)
            z_out[:,nt] += Δt*A*z[:,j]*(t-Δt*j)^(α-1)/gamma(α)
        end 
    end
    return z_out
end

z = zeros(d,Nt)
uT_approx = f(A,z,u0,T,α)
uT
uT_approx[:,end] # done, they are the same!

function fixed_point(g::Function,z0,rtol=1e-4,max_iter =100)
    g0 = g(z0)
    iter=0
    tol = Inf
    res = zeros(0)
    while iter < max_iter && tol > rtol
        z0 = g0
        g0 = g(z0)
        tol = norm(g0-z0)/norm(g0)
        append!(res,tol)
        iter += 1
    end
    return g0, res
end

g(z0) = g(A,z0,u0,T,α) 
z0_eq, res = fixed_point(g,zeros(d,Nt))
z0_eq

res