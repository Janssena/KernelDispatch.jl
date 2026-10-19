# Locating the jax-mps Metal PJRT plugin, for the tests and benchmarks that need a Reactant client
# that actually traces to a GPU.
#
# Reactant's default client is CPU, and `backend_of` answers `CPUBackend()` for a traced array under
# it, so an op call takes the host path and the device-kernel route is untested unless a real Metal
# client is installed first. Anything exercising that path has to build this client explicitly.
#
# Set `JAX_MPS_PLUGIN` to the plugin's `libpjrt_plugin_mps.dylib`; without it the traced tests skip.
#
# No side effects on include: `mps_client()` is a function, and returns `nothing` rather than
# throwing when the plugin is absent, so callers skip cleanly on a machine without it.

const JAX_MPS_PLUGIN = get(ENV, "JAX_MPS_PLUGIN", "")      # "" -> no plugin: `mps_client()` is `nothing`

# MLX (the engine behind the plugin) reserves ~1.5x the working set and does not release freed
# buffers between unrelated compiles. Must be set before the plugin creates its client.
get!(ENV, "JAX_MPS_CACHE_LIMIT_BYTES", string(4 * 1024^3))

"""
    mps_client() -> Reactant PJRT client or `nothing`

The jax-mps Metal client, or `nothing` if the plugin is not installed or fails to load. Cached, so
repeated calls in one process reuse the one client — PJRT clients are not free, and two live clients
for the same device is not a supported configuration.
"""
const _MPS_CLIENT = Ref{Any}(missing)

function mps_client()
    _MPS_CLIENT[] === missing || return _MPS_CLIENT[]
    _MPS_CLIENT[] = if !isfile(JAX_MPS_PLUGIN)
        nothing
    else
        try
            Reactant.XLA.PJRT.MakeClientUsingPluginAPI(JAX_MPS_PLUGIN, "mps", "MPS")
        catch
            nothing
        end
    end
    return _MPS_CLIENT[]
end
