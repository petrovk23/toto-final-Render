#ifndef ANALYSIS_ENGINE_H
#define ANALYSIS_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This struct represents a single result item from the "top" or "chain" analysis.
 * We'll return an array of these from the C engine.
 */
typedef struct {
    // For single or top analysis:
    // Combination in ascending order as a string, e.g. "1,2,3,4,5,6"
    char combination[256];

    // The average rank across subsets
    double avg_rank;

    // The min value across subsets
    double min_value;

    // Subsets (for demonstration, just a truncated string)
    char subsets[512];

    // For chain analysis only:
    // If l = -1, we store these extra fields
    int draw_offset;
    int draws_until_common;
    int analysis_start_draw;
    int is_chain_result; // 1 if chain result, 0 if normal
} AnalysisResultItem;

/*
 * Main function to run analysis.
 *   game_type: e.g. "6_42"
 *   draws: array of 6-element arrays containing the draws. draws[i][0..5]
 *   draws_count: how many draws we have
 *   j, k, m, l, n, last_offset: same meaning as in your Python code
 *
 * returns: a pointer to an array of AnalysisResultItem. 'out_len' is set
 *          to the number of results. The caller (Python) will free it
 *          with free_analysis_results().
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

/*
 * Frees the array returned by run_analysis_c().
 */
void free_analysis_results(AnalysisResultItem* results);

#ifdef __cplusplus
}
#endif

#endif
