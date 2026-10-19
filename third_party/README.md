# Vendored CLI dependencies

Only the CLI output/metadata layer uses these pinned, unmodified headers. The
solver library and CUDA kernels do not depend on them. Building needs no network
fetch or new system package. `sources.json` records download URLs, sizes, and
SHA-256 hashes for every header and license.

| Component | Version | Use | License |
| --- | --- | --- | --- |
| [nlohmann/json](https://github.com/nlohmann/json/releases/tag/v3.12.0) | 3.12.0 | Strict JSON parsing and serialization | [MIT](nlohmann/LICENSE.MIT) |
| [xxHash](https://github.com/Cyan4973/xxHash/releases/tag/v0.8.3) | 0.8.3 | XXH3-128 streaming/file fingerprints | [BSD 2-Clause](xxhash/LICENSE) |

XXH3-128 is a noncryptographic integrity fingerprint. Release/scientific gates
continue to use SHA-256. These notices cover these two headers only. The project
itself is licensed under MPL-2.0 ([`LICENSE`](../LICENSE)), and the complete
third-party inventory, including vendored OpenCorr and nanoflann, is
[`NOTICE.md`](../NOTICE.md).
