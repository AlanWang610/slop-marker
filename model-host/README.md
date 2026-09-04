# model-host/

Staging area for the files served from the static model host (§6.4). Gitignored.

Contents are a copy of `artifacts/bundles/<version>/`, laid out under the path the
extension fetches. Publishing is: assemble bundle → verify `SHA256SUMS` → sync here →
upload → confirm the served files hash to the same values.

`tools/publish_bundle.py` does all of that against GitHub Releases, with the release tag set
to the model version so the extension's `<base>/<version>/<file>` layout falls out for free.
It dry-runs by default, since `--yes` creates a public release. The final step is the one
that earns the trust: every file is downloaded back from its public URL and hashed, and the
run fails if the *served* bytes disagree with `SHA256SUMS`.

The extension checks every downloaded file against the `SHA256SUMS` it ships with, so a
mismatched upload fails closed rather than scoring with the wrong weights.
