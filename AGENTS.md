# Repository Guidelines

## Project Structure & Module Organization

- `c/` contains the dependency-light engine, optional CUDA/HIP/Metal/Vulkan backends, helpers, and C/Python tests under `c/tests/`.
- `colibri/` is the Python package and CLI; `web/` is the React/Vite dashboard; `desktop/` is the Tauri v2 shell for the shared UI.
- `docs/`, `GPU_BACKENDS.md`, and root `README*.md` files document formats and usage. `assets/`, `site/`, and `docker/` contain static assets and container definitions.
- Keep generated binaries, model files, downloaded checkpoints, and benchmark artifacts out of commits.

## Build, Test, and Development Commands

From the repository root:

```sh
make check                           # portable CPU build plus C and Python tests
make -C c colibri                    # build the engine
make -C c cuda-test CUDA_ARCH=native # validate CUDA changes on a CUDA host
cd web && npm ci && npm run dev      # run the web UI locally
cd web && npm test && npm run build  # test and production-build the UI
```

For the desktop wrapper, install the Tauri CLI, then run `cargo tauri dev` from `desktop`; validate with `cargo fmt --manifest-path src-tauri/Cargo.toml --check` and `cargo check --manifest-path src-tauri/Cargo.toml`.

## Coding Style & Naming Conventions

Use LF endings and final newlines. Follow `.editorconfig`: four spaces for C/C++/CUDA/Python, two for TypeScript/JSON/CSS, and tabs in Makefiles. Apply the LLVM-based `.clang-format` to C-family changes. Use `snake_case` for C/Python names, `PascalCase` for React components, and descriptive `test_*.c`/`test_*.py` names. Preserve the dependency-free default CPU path; GPU backends remain opt-in.

## Testing Guidelines

C tests run with `make -C c test-c`; Python tests use `unittest` discovery via `make -C c test-python`; `make check` runs both. Web tests are `*.test.ts` files beside helpers and run with `npm test`. Add focused regression coverage and run the narrowest relevant suite before the full gate. CUDA/backend changes should include `make -C c cuda-test` where hardware permits.

## Commit & Pull Request Guidelines

Use concise, imperative commit subjects; recent history uses `Fix`, `Mention`, `chore:`, and `fix(release):`. Keep commits focused. Open pull requests against `dev`, describe the problem and smallest solution, link an issue when applicable, and include UI screenshots. Report validation commands, hardware and repeatable measurements for performance claims, and confirm the CPU build remains clean and dependency-free.

## Security & Configuration Tips

Do not commit secrets, model downloads, generated binaries, or benchmark output. Pin external model revisions in download tooling, and document new environment variables or optional GPU dependencies.
