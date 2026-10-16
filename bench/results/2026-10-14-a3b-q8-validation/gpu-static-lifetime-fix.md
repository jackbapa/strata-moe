# Static GPU expert cleanup lifetime correction

The opt-in CUDA static-expert provider can outlive the CUDA backend/context that executed it. Engine and fixture cleanup destroy model contexts, schedulers and backends before releasing the shared static expert registry. Consequently, the provider's saved `last_stream` can already have been destroyed when its destructor runs.

The destructor now selects its CUDA device and calls `cudaDeviceSynchronize()` before freeing per-layer descriptors, counters, mapped host complements and the last shared scratch owner. It no longer dereferences the borrowed stream handle during destruction. Shared scratch release follows that device drain; partial scratch allocation failures have submitted no kernels. Request-boundary statistics still join the saved stream while its executing context is alive.

This correction affects only destruction of opt-in static expert handles. It does not change the kernels, cache placement, arithmetic, normal request submission or default engine path. It adds no device-wide synchronization inside inference.

Existing parity and full-model logs remain untouched and describe the earlier binary. At the time this note was written, the corrected source had been reviewed and its vendor patch regenerated; the next build/runtime validation belongs to the coordinating agent. No GPU command or build was run by the author of this correction.
