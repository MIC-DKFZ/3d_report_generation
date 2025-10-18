# Evaluate Cached Metrics

This directory contains locally cached metric modules from the
[`evaluate`](https://github.com/huggingface/evaluate) library.

These files were copied from a working environment where they were originally downloaded via the Hugging Face Hub.
They are stored here to allow evaluation on compute nodes that do **not have internet access**.

## Source

Downloaded automatically by `evaluate` version 0.4.5 using:
```
evaluate.load("<metric_name>")
```
Copied from the local Hugging Face cache at:
```
~/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--<metric_name>
```

## License

Each metric module retains its **original license** and copyright information.
See the headers in each file or refer to:
- https://github.com/huggingface/evaluate/tree/main/metrics

## Maintenance

If the metrics need updating:
1. Re-download them in an internet-connected environment using the same `evaluate` version.
2. Copy the new cached files into this directory.
3. Update this README with the new version/date.

_Last updated: 2025-10-17_

---

## Local Modifications

### METEOR

The `METEOR` metric in this directory was modified from the original
Hugging Face implementation to return the list of per-example scores
instead of the mean value.

**Original code:**
```python
return {"meteor": np.mean(scores)}
```

**Modified code:**
```python
return {"meteor": scores}
```
