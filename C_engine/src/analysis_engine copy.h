#ifndef ANALYSIS_ENGINE_H
#define ANALYSIS_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char combination[256];     // formatted string of numbers
    double avg_rank;          // average rank
    double min_value;         // minimum rank value
    char subsets[512];        // string representation of subsets
    int draw_offset;          // offset from latest draw
    int draws_until_common;   // draws until common subset found
    int analysis_start_draw;  // starting draw for analysis
    int is_chain_result;      // 1 if chain analysis result
} AnalysisResultItem;

// Main entry point called by Python via ctypes
AnalysisResultItem* run_analysis_c(
    const char* game_type,    // e.g. "6_42" or "6_49"
    int** draws,              // array of draw arrays
    int draws_count,          // total number of draws
    int j,                    // size of combinations to generate
    int k,                    // size of subsets to check
    const char* m,            // "avg" or "min" sort mode
    int l,                    // number of top combinations (-1 for chain)
    int n,                    // number of non-overlapping combinations
    int last_offset,          // offset from latest draw
    int* out_len             // number of results returned
);

// Free the results array
void free_analysis_results(AnalysisResultItem* results);

#ifdef __cplusplus
}
#endif

#endif
