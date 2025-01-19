#ifndef ANALYSIS_ENGINE_H
#define ANALYSIS_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

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
    char subsets[65536];

    // For chain analysis:
    //   draw_offset: "Analysis #"
    //   draws_until_common: "Top-Ranked Duration"
    //   analysis_start_draw: "For Draw" ( = total_draws - offset )
    //   is_chain_result: 1 if from chain analysis
    int draw_offset;
    int draws_until_common;
    int analysis_start_draw;
    int is_chain_result;
} AnalysisResultItem;

/**
 * run_analysis_c(...)
 * -------------------
 * Main entry point for both standard (l >= 1) and chain (l == -1) analyses.
 *
 * If l != -1, returns up to l + n combos (the top-l plus optional “selected” combos),
 * but the “selected” combos are now chosen to avoid overlapping *k*-subsets with each other.
 *
 * If l == -1, runs the “chain” of repeated top-1 analyses:
 *   - Each iteration uses the current offset.
 *   - After each top-1 result, searches forward draws for a common k-subset.
 *     If never found, we imagine a future draw to finalize “Top-Ranked Duration.”
 *
 * Caller must free the returned pointer with free_analysis_results(...).
 *
 * out_len is set to the number of AnalysisResultItem results.
 *
 * Returns NULL if no results (out_len=0).
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
    int* out_len
);

/**
 * free_analysis_results(...)
 * --------------------------
 * Frees the array returned by run_analysis_c.
 */
void free_analysis_results(AnalysisResultItem* results);

#ifdef __cplusplus
}
#endif

#endif
