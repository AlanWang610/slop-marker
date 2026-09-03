# artifacts/

Training and export outputs. Gitignored.

```
runs/<run-id>/        checkpoints, logs, config snapshot, metrics
export/<run-id>/      fp32 ONNX, int8 ONNX, parity report vs PyTorch (§5.3)
bundles/<version>/     model.onnx, tokenizer.json, tokenizer_config.json, config.json,
                      calibration.json, SHA256SUMS  (§5)
```

A bundle directory is the unit of release. It is assembled once, checksummed, and never
edited in place — a change means a new version string, which invalidates cached scores
on both browsers (§4.7, §11).
