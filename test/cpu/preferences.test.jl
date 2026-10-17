# `@default_kernel` in a real package (SDPAOps): the default comes from a compile-time preference, and
# changing the preference re-precompiles the package with the new default baked in. Each step runs in
# a fresh Julia process, with its own environment and a depot of its own in front for the compiled
# caches.

using Test, KernelDispatch, SDPAOps

# Under `Pkg.test` the parent has `JULIA_LOAD_PATH=@:<sandbox>`, which has no `@stdlib`: a child
# inheriting it cannot even `using Pkg`. A child gets the default load path and its own project.
function child(code, env; depots=nothing)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$env -e $code`
    cmd = addenv(cmd, "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
    depots === nothing || (cmd = addenv(cmd, "JULIA_DEPOT_PATH" => depots))
    return cmd
end

function pref_env()
    env, depot = mktempdir(), mktempdir()
    code = "using Pkg; Pkg.offline(true); Pkg.develop([PackageSpec(path=$(repr(pkgdir(KernelDispatch)))), " *
           "PackageSpec(path=$(repr(pkgdir(SDPAOps))))])"
    run(pipeline(child(code, env); stdout=devnull))
    return env, depot
end

# Run `code` after `using SDPAOps, KernelDispatch` in the environment; (stdout, stderr).
function in_env(env, depot, code)
    out, err = IOBuffer(), IOBuffer()
    depots = join([depot; DEPOT_PATH], Sys.iswindows() ? ";" : ":")
    run(pipeline(child("using SDPAOps, KernelDispatch; $code", env; depots); stdout=out, stderr=err))
    return String(take!(out)), String(take!(err))
end

@testset "preference-backed default" begin
    env, depot = pref_env()
    current = "print(current_kernel(:sdpa))"

    @test first(in_env(env, depot, current)) == "flash"                       # no preference: first candidate
    @test first(in_env(env, depot, "use_kernel!(:sdpa, :ka); $current")) == "ka"       # session only
    @test first(in_env(env, depot, current)) == "flash"                       # not persisted

    in_env(env, depot, "set_default_kernel!(:sdpa, :ka)")
    @test occursin("kernel.sdpa", read(joinpath(env, "LocalPreferences.toml"), String))
    # A compile-time preference: the cache is rebuilt and `:ka` is the baked-in default.
    @test first(in_env(env, depot, current)) == "ka"
    @test first(in_env(env, depot, "print(KernelDispatch.default_kernel(SDPAOps.SDPA()))")) == "SDPAKA()"

    in_env(env, depot, "clear_default_kernel!(:sdpa)")
    @test first(in_env(env, depot, current)) == "flash"

    # A preference naming no candidate warns (while precompiling) and falls back.
    write(joinpath(env, "LocalPreferences.toml"), "[SDPAOps]\n\"kernel.sdpa\" = \"nope\"\n")
    out, err = in_env(env, depot, current)
    @test out == "flash"
    @test occursin("not a default candidate", err)
    @test_throws Exception in_env(env, depot, "set_default_kernel!(:sdpa, :nope)")
end
