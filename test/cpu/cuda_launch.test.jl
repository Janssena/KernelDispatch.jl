# The traced CUDA launch, as far as it can be checked without a GPU: the protobuf KernelDispatch sends to
# XLA's `triton_kernel_call` handler (decoded here by a small independent reader), the zlib wrapping, the
# argument-order and size restrictions, and the shape of the emitted `stablehlo.custom_call`. That the
# handler accepts it and launches the kernel is checked on a CUDA machine (test/cuda_probe.jl, gpu/reactant).

using Test, KernelDispatch, Reactant, Zlib_jll

const RXC = Base.get_extension(KernelDispatch, :KernelDispatchReactantExt)
isdefined(@__MODULE__, :reverse_symbol) || include(joinpath(@__DIR__, "..", "ir_utils.jl"))

# --- a minimal protobuf reader: field => [values], length-delimited values as bytes ----------------------
function pb_read(data::AbstractVector{UInt8})
    fields = Dict{Int,Vector{Any}}()
    i = 1
    function varint()
        x, shift = UInt64(0), 0
        while true
            b = data[i]; i += 1
            x |= UInt64(b & 0x7f) << shift
            b & 0x80 == 0 && return x
            shift += 7
        end
    end
    while i <= length(data)
        tag = varint()
        field, wire = Int(tag >> 3), Int(tag & 7)
        v = if wire == 0
            varint()
        elseif wire == 2
            n = Int(varint()); r = data[i:i+n-1]; i += n; r
        else
            error("unexpected wire type $wire")
        end
        push!(get!(fields, field, Any[]), v)
    end
    return fields
end

zlib_uncompress(data::Vector{UInt8}, n::Int) = begin
    out = Vector{UInt8}(undef, n); len = Ref(Csize_t(n))
    rc = ccall((:uncompress, Zlib_jll.libz), Cint, (Ptr{UInt8}, Ptr{Csize_t}, Ptr{UInt8}, Culong),
        out, len, data, length(data))
    rc == 0 || error("zlib uncompress failed ($rc)")
    return out[1:len[]]
end

@testset "zlib stored blocks" begin
    for n in (0, 1, 100, 65535, 65536, 200_000)
        data = rand(UInt8, n)
        @test zlib_uncompress(RXC._zlib_stored(data), n) == data
    end
end

@testset "TritonAnyKernelCall encoding" begin
    image = UInt8[0x7f, 0x45, 0x4c, 0x46, 1, 2, 3]
    params = [(:array, 0), (:array, 64), (:scalar, collect(reinterpret(UInt8, [2.5f0]))),
              (:scalar, collect(reinterpret(UInt8, [Int32(-3)]))), (:scalar, collect(reinterpret(UInt8, [7]))),
              (:scalar, UInt8[1])]
    msg = RXC._kernel_call_proto(; entry="scale", image, is_ptx=false, num_warps=2, shared_mem=128,
        grid=(3, 1, 1), params, name="scale")
    any_call = pb_read(msg)
    @test String(only(any_call[4])) == "scale"
    call = pb_read(only(any_call[1]))
    @test only(call[2]) == 3 && only(call[3]) == 1 && only(call[4]) == 1
    kernel = pb_read(only(call[1]))
    @test String(only(kernel[1])) == "scale"
    @test only(kernel[2]) == 2 && only(kernel[3]) == 128
    mi = pb_read(only(kernel[11]))
    @test only(mi[1]) == image && String(only(mi[2])) == "scale"
    @test !haskey(kernel, 4)                                                    # no PTX when there is an image
    ps = [pb_read(p) for p in call[5]]
    @test length(ps) == 6
    @test haskey(ps[1], 1) && haskey(ps[2], 1)
    @test only(pb_read(only(ps[2][1]))[1]) == 64                                # bytes_to_zero
    @test only(ps[3][4]) == reinterpret(UInt32, 2.5f0)                          # a 4-byte scalar: its bits, as u32
    @test only(ps[4][4]) == reinterpret(UInt32, Int32(-3))
    @test only(ps[5][6]) == 7                                                   # 8 bytes: u64
    @test only(ps[6][2]) == 1                                                   # 1 byte: bool

    ptx = RXC._kernel_call_proto(; entry="k", image=codeunits(".version 7.0"), is_ptx=true, num_warps=1,
        shared_mem=0, grid=(1, 1, 1), params=[], name="k")
    k = pb_read(only(pb_read(only(pb_read(ptx)[1]))[1]))
    @test String(only(k[4])) == ".version 7.0" && !haskey(k, 11)

    @test_throws ArgumentError RXC._kernel_call_proto(; entry="k", image, is_ptx=false, num_warps=1,
        shared_mem=0, grid=(1, 1, 1), params=[(:scalar, UInt8[1, 2])], name="k")   # 2 bytes: no such kind
end

@testset "buffer order and layout" begin
    shapes, types = [(4,)], (Float32,)
    ok = RXC._cuda_params([(:in, 0), (:in, 1), (:out, 0), (:bytes, "[0,0,0,64]")], [(4,), (4,)][1:1], types)
    @test [k for (k, _) in ok] == [:array, :array, :array, :scalar]
    zeroed = RXC._cuda_params([(:in, 0), (:out_zero, 0)], shapes, types)
    @test zeroed[2] == (:array, 16)                                             # 4 Float32 elements
    # An output before an input cannot be expressed: XLA hands over operands, then results.
    @test_throws ArgumentError RXC._cuda_params([(:out, 0), (:in, 0)], shapes, types)
    @test_throws ArgumentError RXC._cuda_params([(:in, 1), (:in, 0), (:out, 0)], shapes, types)
end

@testset "custom_call(::CUDABackendTag) emission" begin
    cubin = tempname() * ".cubin"
    write(cubin, UInt8[0x7f, 0x45, 0x4c, 0x46, 0, 1, 2, 3])
    tag = KernelDispatch.CUDABackendTag()
    layout = [(:in, 0), (:out, 0), (:bytes, "[0,0,0,64]")]
    function f(x)
        y, = KernelDispatch.custom_call(tag, "scale", cubin, (x,), [(4,)], layout;
            grid=(1, 1, 1), threadgroup=64, result_eltypes=(Float32,))
        return y
    end
    # Default: the typed-FFI form, a dictionary with the payload in `opaque`. Reactant's raising pass
    # (`raise_triton_custom_call`) only matches the legacy name, so it leaves this one alone.
    x = Reactant.to_rarray(rand(Float32, 4))
    ir = repr(Reactant.@code_hlo optimize = false f(x))
    @test occursin("triton_kernel_call_ffi", ir) && occursin("api_version = 4", ir) && occursin("opaque", ir)

    # The legacy convention: a string and api_version 2, which that pass would rewrite (and reject, there being
    # no Triton IR in the payload) unless it is switched off.
    withenv("KERNELDISPATCH_CUDA_CUSTOM_CALL" => "triton_kernel_call@2") do
        off = Reactant.CompileOptions(; raise_triton_custom_call=false)
        ir = repr(Reactant.@code_hlo optimize = false compile_options = off f(x))
        @test occursin("@triton_kernel_call(", ir) && occursin("api_version = 2", ir)
        @test_throws Exception Reactant.@code_hlo optimize = false f(x)
    end

    emit(; kw...) = KernelDispatch.custom_call(tag, "scale", cubin, (), [(4,)], layout;
        grid=(1, 1, 1), threadgroup=64, kw...)
    err(f) = try f(); nothing catch e; e end
    # What cannot be launched this way says so, instead of failing inside XLA.
    e = err(() -> emit(is_ka=true))
    @test e isa ArgumentError && occursin("KernelAbstractions", e.msg)
    e = err(() -> KernelDispatch.custom_call(tag, "scale", cubin, (), [(4,)], layout; grid=(1, 1, 1), threadgroup=48))
    @test e isa ArgumentError && occursin("multiple of 32", e.msg)
    e = err(() -> KernelDispatch.custom_call(tag, "scale", "/nonexistent.cubin", (), [(4,)], layout;
        grid=(1, 1, 1), threadgroup=64))
    @test e isa ArgumentError && occursin("not found", e.msg)
end
