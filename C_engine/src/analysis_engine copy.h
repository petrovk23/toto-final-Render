#ifndef ANALYSIS_ENGINE_H
#define ANALYSIS_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Callback function signature for reporting progress to Python.
 * - processed: how many combos have been processed so far
 * - total: total combos to process
 * - user_data: pointer back to Python state
 */
typedef void (*progress_callback_t)(int processed, int total, void* user_data);

/**
 * Callback function signature for checking cancellation from Python.
 * If this returns nonzero, the C code should abort the analysis immediately.
 */
typedef int (*cancel_callback_t)(void* user_data);

/**
 * AnalysisResultItem
 * ------------------
 * Holds a single analysis result, whether from standard top-l
 * or from chain analysis (l == -1).
 */
typedef struct {
    char combination[256];
    double avg_rank;
    double min_value;
    char subsets[512];

    // For chain analysis:
    //   draw_offset: "Analysis #"
    //   draws_until_common: "Top-Ranked Duration"
    //   analysis_start_draw: "For Draw"
    //   is_chain_result: 1 if from chain analysis
    int draw_offset;
    int draws_until_common;
    int analysis_start_draw;
    int is_chain_result;
} AnalysisResultItem;

/**
 * run_analysis_c(...):
 * Main entry point for both standard (l >= 1) and chain (l == -1) analyses.
 *
 * If l != -1, returns up to l + n combos (the top-l plus optional “selected” combos).
 * If l == -1, runs a “chain” approach.
 *
 * Caller must free the returned pointer with free_analysis_results(...).
 *
 * out_len is set to the number of AnalysisResultItem results.
 *
 * progress_cb (optional): function to be called with partial progress.
 * cancel_cb (optional): function to be called to check if the user cancelled.
 * user_data: pointer passed back to these callbacks.
 *
 * Returns NULL if no results (out_len=0) or if it was cancelled early.
 */
AnalysisResultItem* run_analysis_c(
    const char* game_type,
    int** draws,
    int draws_count,
    int j,
    int k,
    const char* m,
    int l,
    int n,
    int last_offset,
    int* out_len,
    progress_callback_t progress_cb,
    cancel_callback_t cancel_cb,
    void* user_data
);

/**
 * free_analysis_results(...):
 * Frees the array returned by run_analysis_c.
 */
void free_analysis_results(AnalysisResultItem* results);

#ifdef __cplusplus
}
#endif

#endif
