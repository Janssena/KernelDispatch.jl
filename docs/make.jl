using Documenter
using KernelDispatch

DocMeta.setdocmeta!(KernelDispatch, :DocTestSetup, :(using KernelDispatch); recursive=true)

makedocs(;
    modules = [KernelDispatch],
    sitename = "KernelDispatch.jl",
    authors = "Janssena",
    format = Documenter.HTML(;
        canonical = "https://janssena.github.io/KernelDispatch.jl",
        edit_link = "main",
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Switching kernels" => "tutorials/1-switching-kernels.md",
            "Registering kernels" => "tutorials/2-registering-kernels.md",
            "Forward, backward and Enzyme" => "tutorials/3-forward-backward-enzyme.md",
            "Triton to .metallib" => "tutorials/4-triton-metallib.md",
        ],
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/Janssena/KernelDispatch.jl.git",
    devbranch = "main",
    push_preview = true,
)
