# Warmup v102: unsupported host, no runtime execution

Source `4490ef427b987b62334dad32a6e40ab5b22bc98a`. Both devices are physical
Tesla T4, but `cudaDeviceCanAccessPeer` returns allowed=0 in both directions.
The existing admission returns UNSUPPORTED_HOST before dependency/build
work. This is neither a warmup pass nor a warmup-runtime failure.
Small summary/inventory logs retained at `build/kaggle-v102-observation`.

Same immutable workload/source was resubmitted as v103 after retention.
The other notebook v5 remained RUNNING and was not overwritten. At the
latest status check both v103 and v5 are RUNNING; no third notebook exists.
