# Where the recorded Triton / trifast outputs live. They are real kernels compiled on an NVIDIA machine (tens of MB), so they are
# not part of the repository's working tree but an ARTIFACT (test/Artifacts.toml): downloaded the first time a test needs them,
# into Julia's artifact cache, and never when the package is only used. This file says where to look, in order:
#
#   1. `KD_FIXTURES=/some/dir`, a directory that holds `triton/` and/or `trifast/` (the layout of the artifact);
#   2. test/fixtures/triton and test/fixtures/trifast, if they exist: what `make_triton_fixture.py` and `make_trifast_fixture.py`
#      write, so freshly generated fixtures are used before anything is downloaded;
#   3. the artifact `kernel_dispatch_fixtures` of test/Artifacts.toml (made by test/fixtures/make_artifact.jl), downloaded on first use.
#
# A fixture directory that none of these provides is `""`: the tests that need it are skipped with a warning, not failed.

using Pkg.Artifacts: ensure_artifact_installed

const FIXTURE_ARTIFACT = "kernel_dispatch_fixtures"

# The artifact's root directory, downloaded if need be; `nothing` when there is no Artifacts.toml (yet) or it cannot be fetched.
const _ARTIFACT_ROOT = Ref{Any}(missing)
function fixture_artifact_root()
    _ARTIFACT_ROOT[] === missing || return _ARTIFACT_ROOT[]
    toml = joinpath(@__DIR__, "Artifacts.toml")
    root = if !isfile(toml)
        @info "no test/Artifacts.toml: the recorded Triton fixtures are not published yet (make them with test/fixtures/make_artifact.jl)"
        nothing
    else
        try
            ensure_artifact_installed(FIXTURE_ARTIFACT, toml)
        catch err
            @warn "could not get the fixtures artifact `$FIXTURE_ARTIFACT` (offline? set KD_FIXTURES to a directory that has them)" exception = (err, catch_backtrace())
            nothing
        end
    end
    return _ARTIFACT_ROOT[] = root
end

"""
    fixture_dir(kind) -> String

The directory `kind` (`"triton"` or `"trifast"`) of the recorded fixtures, or `""` if there is none (see the top of this file).
"""
function fixture_dir(kind::AbstractString)
    env = get(ENV, "KD_FIXTURES", "")
    for root in (env, joinpath(@__DIR__, "fixtures"))
        isempty(root) && continue
        isdir(joinpath(root, kind)) && return joinpath(root, kind)
    end
    root = fixture_artifact_root()
    root !== nothing && isdir(joinpath(root, kind)) && return joinpath(root, kind)
    return ""
end
