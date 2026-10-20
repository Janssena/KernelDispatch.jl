# Signatures: the argument types of a binary

```@meta
CurrentModule = KernelDispatch
```

A compiled binary takes raw pointers and bits. Nothing in it notices a `Float16` array bound to a `float*`, or an `Int64`
where the kernel reads four bytes: it runs, and computes something else. So a [`KernelBinary`](@ref) carries a
**signature**, the argument types it was compiled for, and every launch is checked against it. This page says what a signature
contains, in which order, how to give one, and what you see when it does not match.

## What a signature lists

One entry for each argument [`call_binary`](@ref) binds, **in the order it binds them**: the arguments you pass to
`call_binary` (outputs included, where you put them), followed by the binary's [`extras`](@ref). Each entry is a Julia type:

| the argument | its entry |
|:--|:--|
| an array (`Array`, `MtlArray`, `CuArray`, a traced array) | `Vector{T}`: only the element type `T` counts, not the dimensions or the array type |
| `OutArray(T, dims...)` | `Vector{T}` |
| `InOutArray(x)` | `Vector{eltype(x)}` |
| a number | its own type, **exactly**: `Float32`, `Int32`, `UInt32`, `Bool`, ... |
| a `Val` | nothing: a `Val` is compiled in and takes no slot |

Two things follow that catch people out:

- **A scalar is checked by exact type.** `Int` is `Int64` on every 64-bit machine, so `size(q, 1)` does not fit an `Int32`
  slot; convert it (`Int32(size(q, 1))`). A bare `2.0` is a `Float64` and does not fit a `Float32` slot.
- **You write the type you pass, not the type in the kernel's source.** A Metal `constant uint& n` is declared `Int32` (or
  `UInt32`, whichever `forward` passes), a `device const float*` is `Vector{Float32}`.

What the launcher adds on its own is not part of the signature: a KernelAbstractions binary's leading context words
(`abi=:ka`) and a Triton binary's hidden trailing scratch pointers are bound for you.

### An example

The `forward` of the [registering tutorial](tutorials/2-registering-kernels.md) binds

```julia
call_binary(op, kern, :fwd, q, k, v, out, n, m, d, dv, scale; extent=n)
```

with `q, k, v` arrays of `Float32`, `out = OutArray(Float32, dv, n)`, four `Int32` sizes and a `Float32` scale. Its signature is

```julia
signature = (Vector{Float32}, Vector{Float32}, Vector{Float32}, Vector{Float32},   # q, k, v, out
             Int32, Int32, Int32, Int32,                                           # n, m, d, dv
             Float32)                                                              # scale
```

For a half-precision binary the buffers change and the scalars stay: `Vector{Float16}` (or `Vector{BFloat16}`) four times,
and still a `Float32` scale.

## Where it comes from

| binary | signature |
|:--|:--|
| [`ka_compile`](@ref) | **recorded for you**, from the `argtypes` you compiled with (the `Val`s dropped) |
| a prebuilt `.metallib` / `.cubin` / `.ptx`: `KernelBinary(...; signature=(...))` | **declared by you**, as above |
| [`triton_binary`](@ref)`(...; signature=(...))` | **declared by you**: the arguments the compiled kernel keeps, as you bind them (see below) |

`signature` has three states:

| `signature=` | meaning |
|:--|:--|
| a tuple of types | checked on every launch |
| not given (`nothing`) | unknown: **not checked**, and the first launch of the binary warns once |
| `false` | declared unchecked, silently |

It is saved with the binary in the kernel's manifest, so a registered binary keeps its signature across sessions.
`KernelBinary(b; kwargs...)` copies a binary with some fields changed and keeps the signature, which rebuilding it field by
field does not.

At a single launch, `check_types=false` (a keyword of `call_binary`, `run_binary`, `prepare_launch`) skips the check.

## What you see

A mismatch throws an `ArgumentError` that names the argument and both types, and lists what the binary takes next to what this
launch binds (`f32` is a buffer of `Float32`, `s:i32` a scalar `Int32`):

```
ArgumentError: argument 1 of binary `scale` is a buffer of f16, but the binary was compiled for a buffer of f32;
it takes (f32, f32, s:f32, s:i32) and this launch binds (f16, f32, s:f32, s:i32). Pass `check_types=false` to launch anyway.

ArgumentError: argument 4 of binary `scale` is a scalar of type i64, but the binary was compiled for a scalar of type i32;
it takes (f32, f32, s:f32, s:i32) and this launch binds (f32, f32, s:f32, s:i64). Pass `check_types=false` to launch anyway.
```

A count that differs says so ("takes 4 arguments ... but this launch binds 3"). The kernel does not run.

A binary with no signature on a device is not checked, and its first launch prints the signature that fits **that launch**:

```
Warning: Binary `scale` (scale.ptx, abi `:explicit`) has no recorded signature, so the types of its arguments are not checked.
This launch binds (Vector{Float16}, Vector{Float16}, Float32, Int32); if that is what the binary was compiled for, pass
`signature=(Vector{Float16}, Vector{Float16}, Float32, Int32)` to `KernelBinary` (or `triton_binary`), or `signature=false`
to silence this. `Val`s take no slot and are left out.
```

Check that this is really what the binary was built for before pasting it: it is the types you *passed*, which is what the
check is for only if that launch was right. The warning is once per file and entry; launches with host arrays never warn.

## Triton

Triton drops `constexpr` arguments and integers equal to 1 from the compiled kernel, and newer versions (3.8.0 has two) add
hidden trailing scratch pointers. The signature is the arguments **you bind**, in the compiled kernel's order, so it excludes the
constexprs and the hidden pointers that `triton_binary` supplies:

```julia
bin = triton_binary("scale_kernel.json"; signature=(Vector{Float32}, Vector{Float32}, Float32, Int32))   # x, y, a, n
```

A Triton `fp32` scalar is a `Float32`, `i32` an `Int32`, and a `*fp16` pointer is a buffer of `Float16`. See the
[Triton tutorial](tutorials/4-triton-kernels.md) for how the compiled argument list is read from the kernel.

## Behind the entries

Internally an entry is a token: `f32`, `f16`, `bf16`, `i32`, `u32`, `bool`, ... for the common types, and `s:` in front of a
scalar's token (`s:f32`), which is how a `Float32` scalar differs from a `Float32` buffer. Any other type is spelled out
with its module and parameters (`Base.Complex{f32}`), so `ComplexF32` and `ComplexF64`, or two packages' `Params`, are
different. [`ka_compile`](@ref) puts the same tokens in the file name (`_f32x4_i32sx4_f32s_v64`: four `Float32` buffers,
four `Int32` scalars, a `Float32` scalar, a `Val(64)`), so builds of one kernel at different types never share a file.

## What is not checked

- The **order and meaning** of arguments of the same type: two `Float32` buffers swapped pass. Pin the order with
  [`record_bindings`](tutorials/2-registering-kernels.md#Checking-what-a-binary-gets).
- **Shapes and strides**: the element type is checked, not the length of the array.
- A binary **without a signature** (it warns once).
- **Triton's value-dependent specialisation**: a binary is valid only for the integer arguments in the same classes (equal to
  1, divisible by 16) as when it was compiled. See the [Triton tutorial](tutorials/4-triton-kernels.md).
- The signature **inside the file**: it is not read from the binary (a cubin or PTX does not say what a pointer points to), so
  a wrong declaration is believed.
