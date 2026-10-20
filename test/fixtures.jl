# The fixture ops of the test suite. SDPAOps (examples/SDPAOps) is the realistic op and carries the
# integration tests; these two small ops exist for what it structurally cannot show, each a
# KernelAbstractions kernel written to the binary contract (raw device pointers, 1-based linear
# indices, bits scalars, `Val`s for compile-time parameters):
#
#   Residual    `z = a·x + y`, a scaled residual connection, with a forward AND a backward *binary*
#               (SDPAOps' backward is plain Julia). Several kernels, so the registry, selection and
#               manifest tests have something to switch between, and fake binaries, so those run
#               without a device.
#   LogitScale  `y = a·x` into a longer, zero-filled output: the binary-level launch features,
#               `OutArray(...; zero=true)`, a kernel with two items per thread, multi-dimensional
#               `extent`, `ka_compile`, an unregistered binary.
#
# SDPAOps' cache is redirected too, so no test writes into the user's cache.

using KernelDispatch, KernelAbstractions
import KernelDispatch: opname, kernelname, host, forward, backward, variant_key, extras, OutArray

# --- Residual ---------------------------------------------------------------------------------------

@kernel unsafe_indices = true function residual_fwd!(z, x, y, a::Float32, n::Int32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    if i <= n
        unsafe_store!(z, a * unsafe_load(x, i) + unsafe_load(y, i), i)
    end
end

@kernel unsafe_indices = true function residual_bwd!(dx, dy, dz, a::Float32, n::Int32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    if i <= n
        g = unsafe_load(dz, i)
        unsafe_store!(dx, a * g, i)
        unsafe_store!(dy, g, i)
    end
end

struct Residual <: AbstractKernelOp end
struct ResidualKA <: AbstractKernel end
# The same binaries without `extras`: what an inferable `call_binary` looks like.
struct ResidualPlain <: AbstractKernel end
const ResidualKernels = Union{ResidualKA,ResidualPlain}

opname(::Residual) = :residual
kernelname(::ResidualKA) = :ka
kernelname(::ResidualPlain) = :plain
host(::Residual, x, y, a) = a .* x .+ y
variant_key(::Residual, ::ResidualKernels, args...) = (nameof(KernelDispatch.realtype(eltype(KernelDispatch._first_array(args)))),)
# The tile size is compiled in: appended as a `Val`, it takes no slot.
extras(::Residual, ::ResidualKA, b::KernelBinary) = (Val(b.params.tg),)

# `:ka` unless switched: a default makes the op call inferred with no `use_kernel!`.
KernelDispatch.@default_kernel Residual ResidualKA() ResidualPlain()

forward(op::Residual, k::ResidualKernels, x, y, a) =
    only(call_binary(op, k, :fwd, OutArray(eltype(x), length(x)), x, y, Float32(a), Int32(length(x));
        extent=length(x)))      # one item per element; the binaries' `tile` is one per thread

function backward(op::Residual, k::ResidualKernels, (x, y, a), z, dz)
    n = length(x)
    dx, dy = call_binary(op, k, :bwd, OutArray(eltype(x), n), OutArray(eltype(y), n), dz, Float32(a), Int32(n);
        extent=n)
    return (dx, dy, nothing)
end

const FIXTURE_CACHE = mktempdir()
KernelDispatch.cache_dir(::Residual) = joinpath(FIXTURE_CACHE, "residual")

"""
Compile both binaries with `ka_compile` and register them as both kernels' variant for `T` (Float32 by default). Selects
nothing: `:ka` is the default, and a switch here would not be visible to the caller's block anyway.
"""
function register_residual!(::Type{T}=Float32; tg=64) where {T}
    be = KernelDispatch.device_backend()
    dir = joinpath(FIXTURE_CACHE, "bin")
    argt = (Vector{T}, Vector{T}, Vector{T}, Float32, Int32, Val{tg})
    fwd = ka_compile(be, residual_fwd!, argt; tg, name="residual_fwd_t$tg", params=(; tg), dir)
    bwd = ka_compile(be, residual_bwd!, argt; tg, name="residual_bwd_t$tg", params=(; tg), dir)
    for k in (ResidualKA(), ResidualPlain())
        register_kernel!(Residual(), k)
        add_variant!(Residual(), k, (nameof(T),);
            fwd, bwd)
    end
end

# --- LogitScale -------------------------------------------------------------------------------------

# `y[i] = a * x[i]` for `i <= n`. The output is allocated LONGER than `n` and zero-initialised, so the
# tail also pins `OutArray(...; zero=true)`.
@kernel unsafe_indices = true function logit_scale_kernel!(y, x, a::Float32, n::Int32, ::Val{TG}) where {TG}
    g = @index(Group, Linear)
    l = @index(Local, Linear)
    i = (g - 1) * TG + l
    if i <= n
        unsafe_store!(y, a * unsafe_load(x, i), i)
    end
end

# The same, each thread covering two consecutive items: a binary whose `tile` is twice its threadgroup.
@kernel unsafe_indices = true function logit_scale2_kernel!(y, x, a::Float32, n::Int32, ::Val{TG}) where {TG}
    t = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    for i in (2t - 1):(2t)
        if i <= n
            unsafe_store!(y, a * unsafe_load(x, i), i)
        end
    end
end

# The same with the input before the output: XLA's kernel-call handler hands a traced CUDA launch its buffers inputs
# first, so this is the KA kernel the traced CUDA tests launch (the others put the output first, as Julia's `f!` does).
@kernel unsafe_indices = true function scale_ins_first!(x, y, a::Float32, n::Int32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    if i <= n
        unsafe_store!(y, a * unsafe_load(x, i), i)
    end
end

"Compile `scale_ins_first!` for the loaded device backend (not registered anywhere): `(x, y, a, n, Val)`."
function compile_scale_ins_first(; tg=LS_TG)
    argtypes = (Vector{Float32}, Vector{Float32}, Float32, Int32, Val{tg})
    return ka_compile(KernelDispatch.device_backend(), scale_ins_first!, argtypes; tg, name="test_scale_ins_first_t$tg")
end

struct LogitScale <: AbstractKernelOp end
struct LogitScaleKA <: AbstractKernel end
struct LogitScaleTwo <: AbstractKernel end      # `logit_scale2_kernel!`: two items per thread
opname(::LogitScale) = :logit_scale
kernelname(::LogitScaleKA) = :ka
kernelname(::LogitScaleTwo) = :two

const LOGIT_CACHE = mktempdir()
KernelDispatch.cache_dir(::LogitScale) = joinpath(LOGIT_CACHE, "logit_scale")
const LS_TG = 64

"Compile the kernel for the loaded device backend (into the default temporary build directory) and register the binary as the op's variant."
function register_logit_scale!(; tg=LS_TG)
    register_kernel!(LogitScale(), LogitScaleKA())
    argtypes = (Vector{Float32}, Vector{Float32}, Float32, Int32, Val{tg})
    bin = ka_compile(KernelDispatch.device_backend(), logit_scale_kernel!, argtypes; tg, name="test_scale_t$tg")
    add_variant!(LogitScale(), LogitScaleKA(), (); main=bin)
end

"`a .* x` through `call_binary`, into a zero-filled output of `out_len` elements."
function logit_scale(x, a; out_len::Int=length(x), tg::Int=LS_TG)
    n = length(x)
    (y,) = call_binary(LogitScale(), LogitScaleKA(), :main,
        OutArray(Float32, out_len; zero=true), x, Float32(a), Int32(n), Val(tg); extent=n)
    return y
end

# --- SDPAOps ----------------------------------------------------------------------------------------

using SDPAOps
const SDPA_CACHE = mktempdir()
KernelDispatch.cache_dir(::SDPAOps.SDPA) = joinpath(SDPA_CACHE, "sdpa")
