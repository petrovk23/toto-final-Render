#ifndef ANALYSIS_ENGINE_H
#define ANALYSIS_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char combination[256];
    double avg_rank;
    double min_value;
    char subsets[512];
    int draw_offset;
    int draws_until_common;
    int analysis_start_draw;
    int is_chain_result;
} AnalysisResultItem;

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

void free_analysis_results(AnalysisResultItem* results);

#ifdef __cplusplus
}
#endif

#endif
