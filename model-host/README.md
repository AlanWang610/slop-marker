# model-host/

Staging area for the files served from the static model host (§6.4). Gitignored.

Contents are a copy of `artifacts/bundles/<version>/`, laid out under the path the
extension fetches. Publishing is: assemble bundle → verify `SHA256SUMS` → sync here →
upload → confirm the served files hash to the same values.

The extension checks every downloaded file against the `SHA256SUMS` it ships with, so a
mismatched upload fails closed rather than scoring with the wrong weights.
