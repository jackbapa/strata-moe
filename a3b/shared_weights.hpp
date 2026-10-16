#pragma once

#include "llama.h"
#include <cstddef>
#include <string>
#include <vector>

namespace strata_a3b {

struct SharedTensor {
    std::string name, type, sha256, target_buffer, previous_draft_buffer;
    size_t bytes = 0;
    bool target_on_host = true;
};
struct SharedWeights {
    std::vector<SharedTensor> tensors;
    size_t reused_gpu_bytes = 0;
    size_t reused_host_bytes = 0;
    size_t retained_draft_host_bytes = 0;
};

// Call after both models load and BEFORE creating the draft context. Load the
// draft's token_embd.weight/output.weight/output_norm.weight into the CPU buffer
// type so their duplicate device storage is avoided. This checks model/file
// types, shapes, byte counts and streamed SHA-256 before borrowing any pointer.
// The target model must outlive the draft model and every draft context.
// Existing CPU allocations remain owned by the draft model; this saves device
// duplication, not the retained CPU copies. No quantization is changed.
SharedWeights share_mtp_weights(llama_model * target, llama_model * draft,
                               const std::string & target_file, const std::string & draft_file);

} // namespace strata_a3b
