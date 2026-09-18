{
  description = "OpenUS dev environment — uv-managed .venv with NixOS CUDA passthrough (GTX 1660 Ti)";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [ uv git libGL glib ];

        env = {
          # nixpkgs unstable has dropped python3.10 (OpenUS pins 3.10 exactly,
          # for numpy==1.24.4 / timm==0.4.12 / the cp310 mamba_ssm wheel), so
          # let uv fetch its own CPython 3.10 build instead of a Nix one.
          # This only works because nix-ld is enabled system-wide (see
          # niriNixOs/modules/features/nix-ld.nix) — it patches the dynamic
          # loader those portable python-build-standalone binaries expect.
          UV_PYTHON_DOWNLOADS = "auto";
          UV_PYTHON = "3.10";
        };

        shellHook = ''
          # torch / mamba_ssm dlopen libcuda.so at runtime. It's deliberately
          # not a Nix package here (the userspace lib must match whatever
          # kernel driver is actually loaded) — /run/opengl-driver/lib is the
          # live path NixOS's nvidia module keeps in sync with the driver.
          # libGL.so.1 comes from the libGL package above: opencv-python
          # (pulled in transitively by pyiqa/clean-fid) links against it even
          # though this is a headless CUDA workload with no display.
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.libGL}/lib:${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
        '';
      };
    };
}
