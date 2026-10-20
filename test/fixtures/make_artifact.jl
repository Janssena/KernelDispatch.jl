# Bundle the recorded Triton / trifast fixtures into the `kernel_dispatch_fixtures` artifact.
#
#     julia test/fixtures/make_artifact.jl [--url URL] [--out DIR] [--fixtures DIR] [--toml FILE]
#
# It takes `triton/` and `trifast/` from the fixtures directory (default: this directory, where make_triton_fixture.py and
# make_trifast_fixture.py write them), makes an artifact of them, archives it as `kernel_dispatch_fixtures.tar.gz` in --out
# (default: the current directory), and writes test/Artifacts.toml with the artifact's content hash, the tarball's sha256 and
# --url, the address the tarball will be downloaded from. Then:
#
#   1. upload the tarball to that address (by default a release asset of this repository: create the release `fixtures-v1`
#      with `gh release create fixtures-v1 kernel_dispatch_fixtures.tar.gz`, or in the GitHub web interface);
#   2. commit test/Artifacts.toml; the fixture directories themselves no longer need to be committed (they are in .gitignore).
#
# The tests (test/fixture_root.jl) use a local test/fixtures/triton and test/fixtures/trifast when they exist, so freshly
# generated fixtures are tested before they are published, and download the artifact (once, into Julia's artifact cache)
# otherwise. A new set of fixtures is a new artifact: bump the release tag in --url, and the hash in Artifacts.toml changes.

using Pkg.Artifacts

function option(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i+1]
end

fixtures = abspath(option("--fixtures", @__DIR__))
out = abspath(option("--out", pwd()))
toml = abspath(option("--toml", joinpath(@__DIR__, "..", "Artifacts.toml")))
url = option("--url", "https://github.com/Janssena/KernelDispatch.jl/releases/download/fixtures-v1/kernel_dispatch_fixtures.tar.gz")
name = "kernel_dispatch_fixtures"

kinds = filter(k -> isdir(joinpath(fixtures, k)), ["triton", "trifast"])
isempty(kinds) && error("no triton/ or trifast/ directory in $fixtures: run make_triton_fixture.py / make_trifast_fixture.py first")

hash = create_artifact() do dir
    for k in kinds
        cp(joinpath(fixtures, k), joinpath(dir, k))
    end
end
mkpath(out)
tarball = joinpath(out, name * ".tar.gz")
sha = archive_artifact(hash, tarball)
bind_artifact!(toml, name, hash; download_info=[(url, sha)], lazy=true, force=true)

println("artifact $name: git-tree-sha1 $hash")
println("  contains: ", join(kinds, ", "), "  ($(round(filesize(tarball) / 2^20; digits=1)) MiB)")
println("  tarball:  $tarball")
println("  sha256:   $sha")
println("  url:      $url")
println("  wrote:    $toml")
println("\nNow upload the tarball to that url, and commit the Artifacts.toml.")
