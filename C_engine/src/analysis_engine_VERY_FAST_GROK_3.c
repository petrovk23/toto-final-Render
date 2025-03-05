#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "analysis_engine.h"

#define MAX_COMBO_STR 255
#define MAX_SUBSETS_STR 65535
#define MAX_ALLOWED_J 200
#define MAX_ALLOWED_OUT_LEN 1000000
#define MAX_NUMBERS 50
#define HASH_SIZE (1 << 24)  // 16M entries

typedef unsigned long long uint64;
typedef unsigned int uint32;

// ----------------------------------------------------------------------
// Internal data and lookups
// ----------------------------------------------------------------------
static uint64 nCk_table[MAX_NUMBERS][MAX_NUMBERS];
static int bit_count_table[256];
static int initialized = 0;

typedef struct {
    uint64* keys;     // Subset bit patterns
    int* values;      // Last occurrence
    int size;
    int capacity;
} SubsetTable;

/**
 * ComboStats holds the enumerated combo stats for top-l searching.
 * 'pattern' is the bit pattern (1<<(num-1)) of the combo's numbers.
 */
typedef struct {
    uint64 pattern;   // Bit pattern of combo
    double avg_rank;
    double min_rank;
    int combo[MAX_NUMBERS];
    int len;
} ComboStats;

// ----------------------------------------------------------------------
// Forward declarations
// ----------------------------------------------------------------------
static void init_tables();
static inline int popcount64(uint64 x);
static SubsetTable* create_subset_table(int max_entries);
static void free_subset_table(SubsetTable* table);
static inline uint32 hash_subset(uint64 pattern);
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value);
static inline int lookup_subset(const SubsetTable* table, uint64 pattern);
static inline uint64 numbers_to_pattern(const int* numbers, int count);
static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table);
static void format_combo(const int* combo, int len, char* out);
static void format_subsets(const int* combo, int j, int k, int total_draws,
                           const SubsetTable* table, char* out);

static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data,
    int use_count,
    int j,
    int k,
    const char* m,
    int l,
    int n,
    int max_number,
    int* out_len
);

static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data,
    int draws_count,
    int initial_offset,
    int j,
    int k,
    const char* m,
    int max_number,
    int* out_len
);

// ----------------------------------------------------------------------
// New functions for backtracking
// ----------------------------------------------------------------------
static double compute_avg_rank(int* S, int j, int k, SubsetTable* table, int total_draws);
static void backtrack(
    int* S,
    int size,
    uint64 current_S,
    double current_min_rank,
    int start_num,
    SubsetTable* table,
    int total_draws,
    int max_number,
    int j,
    int k,
    ComboStats* thread_best,
    int* thread_filled,
    int l,
    const char* m
);

// ----------------------------------------------------------------------
// Implementation
// ----------------------------------------------------------------------
static void init_tables() {
    if (initialized) return;
    memset(nCk_table, 0, sizeof(nCk_table));
    for (int n = 0; n < MAX_NUMBERS; n++) {
        nCk_table[n][0] = 1;
        for (int k = 1; k <= n; k++) {
            nCk_table[n][k] = nCk_table[n-1][k-1] + nCk_table[n-1][k];
        }
    }
    for (int i = 0; i < 256; i++) {
        int c = 0;
        for (int b = 0; b < 8; b++) {
            if (i & (1 << b)) c++;
        }
        bit_count_table[i] = c;
    }
    initialized = 1;
}

static inline int popcount64(uint64 x) {
    return __builtin_popcountll(x);
}

static SubsetTable* create_subset_table(int max_entries) {
    SubsetTable* t = (SubsetTable*)malloc(sizeof(SubsetTable));
    if (!t) return NULL;
    t->size = 0;
    t->capacity = max_entries;
    t->keys = (uint64*)calloc(max_entries, sizeof(uint64));
    t->values = (int*)malloc(max_entries * sizeof(int));
    if (!t->keys || !t->values) {
        free(t->keys);
        free(t->values);
        free(t);
        return NULL;
    }
    for (int i = 0; i < max_entries; i++) {
        t->values[i] = -1;
    }
    return t;
}

static void free_subset_table(SubsetTable* table) {
    if (!table) return;
    free(table->keys);
    free(table->values);
    free(table);
}

static inline uint32 hash_subset(uint64 pattern) {
    return (uint32)(pattern & (HASH_SIZE - 1));
}

static inline void insert_subset(SubsetTable* table, uint64 pattern, int value) {
    uint32 idx = hash_subset(pattern);
    while (1) {
        if (table->values[idx] == -1 || table->keys[idx] == pattern) {
            table->keys[idx] = pattern;
            table->values[idx] = value;
            return;
        }
        idx = (idx + 1) & (HASH_SIZE - 1);
    }
}

static inline int lookup_subset(const SubsetTable* table, uint64 pattern) {
    uint32 idx = hash_subset(pattern);
    while (1) {
        if (table->values[idx] == -1) return -1;
        if (table->keys[idx] == pattern) return table->values[idx];
        idx = (idx + 1) & (HASH_SIZE - 1);
    }
}

static inline uint64 numbers_to_pattern(const int* numbers, int count) {
    uint64 p = 0ULL;
    for (int i = 0; i < count; i++) {
        p |= (1ULL << (numbers[i] - 1));
    }
    return p;
}

static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table) {
    if (k > 6) return;
    int idx[6];
    for (int i = 0; i < k; i++) {
        idx[i] = i;
    }
    while (1) {
        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) {
            pat |= (1ULL << (draw[idx[i]] - 1));
        }
        insert_subset(table, pat, draw_idx);
        int pos = k - 1;
        while (pos >= 0) {
            idx[pos]++;
            if (idx[pos] <= 6 - (k - pos)) {
                for (int x = pos + 1; x < k; x++) {
                    idx[x] = idx[x - 1] + 1;
                }
                break;
            }
            pos--;
        }
        if (pos < 0) break;
    }
}

static double compute_avg_rank(int* S, int j, int k, SubsetTable* table, int total_draws) {
    double sum_ranks = 0.0;
    int count = 0;
    int idx[k];
    for (int i = 0; i < k; i++) idx[i] = i;
    while (1) {
        int subset[k];
        for (int i = 0; i < k; i++) subset[i] = S[idx[i]];
        for (int a = 0; a < k - 1; a++) {
            for (int b = a + 1; b < k; b++) {
                if (subset[a] > subset[b]) {
                    int t = subset[a];
                    subset[a] = subset[b];
                    subset[b] = t;
                }
            }
        }
        uint64 pat = numbers_to_pattern(subset, k);
        int last_seen = lookup_subset(table, pat);
        double rank = (last_seen >= 0) ? (double)(total_draws - last_seen - 1) : (double)total_draws;
        sum_ranks += rank;
        count++;
        int p = k - 1;
        while (p >= 0) {
            if (idx[p] < j - (k - p)) {
                idx[p]++;
                for (int x = p + 1; x < k; x++) {
                    idx[x] = idx[x - 1] + 1;
                }
                break;
            }
            p--;
        }
        if (p < 0) break;
    }
    return sum_ranks / (double)count;
}

static void backtrack(
    int* S,
    int size,
    uint64 current_S,
    double current_min_rank,
    int start_num,
    SubsetTable* table,
    int total_draws,
    int max_number,
    int j,
    int k,
    ComboStats* thread_best,
    int* thread_filled,
    int l,
    const char* m
) {
    if (size == j) {
        double avg_rank = compute_avg_rank(S, j, k, table, total_draws);
        double min_rank = current_min_rank;
        int should_insert = 0;
        if (*thread_filled < l) {
            should_insert = 1;
        } else {
            if (strcmp(m, "avg") == 0) {
                should_insert = (avg_rank > thread_best[l - 1].avg_rank) ||
                                (avg_rank == thread_best[l - 1].avg_rank && min_rank > thread_best[l - 1].min_rank);
            } else {
                should_insert = (min_rank > thread_best[l - 1].min_rank) ||
                                (min_rank == thread_best[l - 1].min_rank && avg_rank > thread_best[l - 1].avg_rank);
            }
        }
        if (should_insert) {
            if (*thread_filled < l) {
                *thread_filled = *thread_filled + 1;
            }
            for (int i = 0; i < j; i++) thread_best[*thread_filled - 1].combo[i] = S[i];
            thread_best[*thread_filled - 1].len = j;
            thread_best[*thread_filled - 1].avg_rank = avg_rank;
            thread_best[*thread_filled - 1].min_rank = min_rank;
            thread_best[*thread_filled - 1].pattern = current_S;
            for (int i = *thread_filled - 1; i > 0; i--) {
                int should_swap = 0;
                if (strcmp(m, "avg") == 0) {
                    should_swap = (thread_best[i].avg_rank > thread_best[i - 1].avg_rank) ||
                                  (thread_best[i].avg_rank == thread_best[i - 1].avg_rank &&
                                   thread_best[i].min_rank > thread_best[i - 1].min_rank);
                } else {
                    should_swap = (thread_best[i].min_rank > thread_best[i - 1].min_rank) ||
                                  (thread_best[i].min_rank == thread_best[i - 1].min_rank &&
                                   thread_best[i].avg_rank > thread_best[i - 1].avg_rank);
                }
                if (should_swap) {
                    ComboStats tmp = thread_best[i];
                    thread_best[i] = thread_best[i - 1];
                    thread_best[i - 1] = tmp;
                } else {
                    break;
                }
            }
        }
        return;
    }
    for (int num = start_num; num <= max_number; num++) {
        if ((current_S & (1ULL << (num - 1))) == 0) {
            S[size] = num;
            uint64 new_S = current_S | (1ULL << (num - 1));
            double min_of_new = total_draws + 1.0;
            if (size >= k - 1) {
                int idx[k - 1];
                for (int i = 0; i < k - 1; i++) idx[i] = i;
                while (1) {
                    int subset[k];
                    for (int i = 0; i < k - 1; i++) subset[i] = S[idx[i]];
                    subset[k - 1] = num;
                    for (int a = 0; a < k - 1; a++) {
                        for (int b = a + 1; b < k; b++) {
                            if (subset[a] > subset[b]) {
                                int t = subset[a];
                                subset[a] = subset[b];
                                subset[b] = t;
                            }
                        }
                    }
                    uint64 pat = numbers_to_pattern(subset, k);
                    int last_seen = lookup_subset(table, pat);
                    double rank = (last_seen >= 0) ? (double)(total_draws - last_seen - 1) : (double)total_draws;
                    if (rank < min_of_new) min_of_new = rank;
                    int p = k - 2;
                    while (p >= 0) {
                        if (idx[p] < size - (k - 1 - p)) {
                            idx[p]++;
                            for (int x = p + 1; x < k - 1; x++) {
                                idx[x] = idx[x - 1] + 1;
                            }
                            break;
                        }
                        p--;
                    }
                    if (p < 0) break;
                }
            }
            double new_min_rank = (current_min_rank < min_of_new) ? current_min_rank : min_of_new;
            if (*thread_filled < l || new_min_rank >= thread_best[l - 1].min_rank) {
                backtrack(S, size + 1, new_S, new_min_rank, num + 1, table, total_draws, max_number, j, k, thread_best, thread_filled, l, m);
            }
        }
    }
}

static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data,
    int use_count,
    int j,
    int k,
    const char* m,
    int l,
    int n,
    int max_number,
    int* out_len
) {
    SubsetTable* table = create_subset_table(HASH_SIZE);
    if (!table) return NULL;
    for (int i = 0; i < use_count; i++) {
        process_draw(&sorted_draws_data[i * 6], i, k, table);
    }

    ComboStats* best_stats = (ComboStats*)malloc(l * sizeof(ComboStats));
    if (!best_stats) {
        free_subset_table(table);
        return NULL;
    }
    memset(best_stats, 0, l * sizeof(ComboStats));

    int filled = 0;
    int error_occurred = 0;

    #pragma omp parallel
    {
        int* S = (int*)malloc(j * sizeof(int));
        ComboStats* thread_best = (ComboStats*)malloc(l * sizeof(ComboStats));
        int thread_filled = 0;
        if (!S || !thread_best) {
            #pragma omp atomic write
            error_occurred = 1;
        } else {
            for (int i = 0; i < l; i++) {
                thread_best[i].avg_rank = -1.0;
                thread_best[i].min_rank = -1.0;
            }
            #pragma omp for schedule(dynamic)
            for (int first = 1; first <= max_number - j + 1; first++) {
                if (!error_occurred) {
                    S[0] = first;
                    uint64 current_S = (1ULL << (first - 1));
                    double current_min_rank = (double)(use_count + 1);
                    backtrack(S, 1, current_S, current_min_rank, first + 1, table, use_count, max_number, j, k, thread_best, &thread_filled, l, m);
                }
            }
            #pragma omp critical
            {
                for (int i = 0; i < thread_filled; i++) {
                    if (filled < l) {
                        memcpy(&best_stats[filled], &thread_best[i], sizeof(ComboStats));
                        filled++;
                    } else {
                        int should_replace = 0;
                        if (strcmp(m, "avg") == 0) {
                            should_replace = (thread_best[i].avg_rank > best_stats[l - 1].avg_rank) ||
                                             (thread_best[i].avg_rank == best_stats[l - 1].avg_rank &&
                                              thread_best[i].min_rank > best_stats[l - 1].min_rank);
                        } else {
                            should_replace = (thread_best[i].min_rank > best_stats[l - 1].min_rank) ||
                                             (thread_best[i].min_rank == best_stats[l - 1].min_rank &&
                                              thread_best[i].avg_rank > best_stats[l - 1].avg_rank);
                        }
                        if (should_replace) {
                            best_stats[l - 1] = thread_best[i];
                            for (int idx = l - 1; idx > 0; idx--) {
                                int should_swap = 0;
                                if (strcmp(m, "avg") == 0) {
                                    should_swap = (best_stats[idx].avg_rank > best_stats[idx - 1].avg_rank) ||
                                                  (best_stats[idx].avg_rank == best_stats[idx - 1].avg_rank &&
                                                   best_stats[idx].min_rank > best_stats[idx - 1].min_rank);
                                } else {
                                    should_swap = (best_stats[idx].min_rank > best_stats[idx - 1].min_rank) ||
                                                  (best_stats[idx].min_rank == best_stats[idx - 1].min_rank &&
                                                   best_stats[idx].avg_rank > best_stats[idx - 1].avg_rank);
                                }
                                if (should_swap) {
                                    ComboStats tmp = best_stats[idx];
                                    best_stats[idx] = best_stats[idx - 1];
                                    best_stats[idx - 1] = tmp;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (S) free(S);
        if (thread_best) free(thread_best);
    }

    if (error_occurred) {
        free(best_stats);
        free_subset_table(table);
        return NULL;
    }

    AnalysisResultItem* results = (AnalysisResultItem*)calloc(l + n, sizeof(AnalysisResultItem));
    if (!results) {
        free(best_stats);
        free_subset_table(table);
        return NULL;
    }

    int top_count = (filled < l) ? filled : l;
    int results_count = 0;

    free_subset_table(table);
    table = create_subset_table(HASH_SIZE);
    for (int i = 0; i < use_count; i++) {
        process_draw(&sorted_draws_data[i * 6], i, k, table);
    }

    for (int i = 0; i < top_count; i++) {
        format_combo(best_stats[i].combo, best_stats[i].len, results[results_count].combination);
        format_subsets(best_stats[i].combo, j, k, use_count, table, results[results_count].subsets);
        results[results_count].avg_rank = best_stats[i].avg_rank;
        results[results_count].min_value = best_stats[i].min_rank;
        results[results_count].is_chain_result = 0;
        results_count++;
    }

    int second_table_count = 0;
    int* pick_indices = NULL;
    if (n > 0 && top_count > 0) {
        pick_indices = (int*)malloc(top_count * sizeof(int));
        memset(pick_indices, -1, top_count * sizeof(int));
        int chosen = 0;
        pick_indices[chosen++] = 0;
        for (int i = 1; i < top_count && chosen < n; i++) {
            uint64 pat_i = best_stats[i].pattern;
            int overlap = 0;
            for (int c = 0; c < chosen; c++) {
                int idxC = pick_indices[c];
                uint64 pat_c = best_stats[idxC].pattern;
                uint64 inter = (pat_i & pat_c);
                if (popcount64(inter) >= k) {
                    overlap = 1;
                    break;
                }
            }
            if (!overlap) {
                pick_indices[chosen++] = i;
            }
        }
        second_table_count = chosen;
    }

    int bottom_start = results_count;
    for (int i = 0; i < second_table_count; i++) {
        int idx = pick_indices[i];
        format_combo(best_stats[idx].combo, best_stats[idx].len, results[bottom_start + i].combination);
        format_subsets(best_stats[idx].combo, j, k, use_count, table, results[bottom_start + i].subsets);
        results[bottom_start + i].avg_rank = best_stats[idx].avg_rank;
        results[bottom_start + i].min_value = best_stats[idx].min_rank;
        results[bottom_start + i].is_chain_result = 0;
    }
    int total_used = results_count + second_table_count;
    *out_len = total_used;

    if (pick_indices) free(pick_indices);
    free_subset_table(table);
    free(best_stats);

    if (total_used == 0) {
        free(results);
        return NULL;
    }
    return results;
}

static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data,
    int draws_count,
    int initial_offset,
    int j,
    int k,
    const char* m,
    int max_number,
    int* out_len
) {
    AnalysisResultItem* chain_results = (AnalysisResultItem*)calloc(initial_offset + 2, sizeof(AnalysisResultItem));
    if (!chain_results) {
        *out_len = 0;
        return NULL;
    }

    uint64* draw_patterns = (uint64*)malloc(draws_count * sizeof(uint64));
    if (!draw_patterns) {
        free(chain_results);
        *out_len = 0;
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        draw_patterns[i] = numbers_to_pattern(&sorted_draws_data[i * 6], 6);
    }

    int chain_index = 0;
    int current_offset = initial_offset;

    while (current_offset >= 0 && current_offset <= draws_count - 1) {
        int use_count = draws_count - current_offset;
        if (use_count < 1) break;

        SubsetTable* table = create_subset_table(HASH_SIZE);
        for (int i = 0; i < use_count; i++) {
            process_draw(&sorted_draws_data[i * 6], i, k, table);
        }

        int* S = (int*)malloc(j * sizeof(int));
        ComboStats best;
        int filled = 0;
        if (S) {
            S[0] = 1;
            uint64 current_S = (1ULL << (S[0] - 1));
            backtrack(S, 1, current_S, (double)(use_count + 1), 2, table, use_count, max_number, j, k, &best, &filled, 1, m);
        }
        free(S);
        free_subset_table(table);

        if (filled == 0) break;

        AnalysisResultItem* out_item = &chain_results[chain_index];
        format_combo(best.combo, best.len, out_item->combination);

        table = create_subset_table(HASH_SIZE);
        for (int i = 0; i < use_count; i++) {
            process_draw(&sorted_draws_data[i * 6], i, k, table);
        }
        format_subsets(best.combo, j, k, use_count, table, out_item->subsets);
        free_subset_table(table);

        out_item->avg_rank = best.avg_rank;
        out_item->min_value = best.min_rank;
        out_item->is_chain_result = 1;
        out_item->draw_offset = chain_index + 1;
        out_item->analysis_start_draw = draws_count - current_offset;

        uint64 combo_pat = best.pattern;
        int i;
        for (i = 1; i <= current_offset; i++) {
            int f_idx = draws_count - 1 - (current_offset - i);
            if (f_idx < 0) break;
            uint64 fpat = draw_patterns[f_idx];
            if (popcount64(combo_pat & fpat) >= k) {
                break;
            }
        }
        if (i > current_offset) i = current_offset + 1;
        out_item->draws_until_common = (i > 0) ? (i - 1) : 0;
        current_offset -= i;
        chain_index++;

        if (current_offset < 0) break;
    }

    free(draw_patterns);
    *out_len = chain_index;
    if (chain_index == 0) {
        free(chain_results);
        return NULL;
    }
    return chain_results;
}

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
) {
    *out_len = 0;
    if (j > MAX_ALLOWED_J) return NULL;
    init_tables();

    int max_number = (strstr(game_type, "6_49")) ? 49 : 42;
    if (draws_count < 1) return NULL;

    int* sorted_draws_data = (int*)malloc(draws_count * 6 * sizeof(int));
    if (!sorted_draws_data) return NULL;
    for (int i = 0; i < draws_count; i++) {
        int temp[6];
        for (int z = 0; z < 6; z++) temp[z] = draws[i][z];
        for (int a = 0; a < 5; a++) {
            for (int b = a + 1; b < 6; b++) {
                if (temp[a] > temp[b]) {
                    int t = temp[a];
                    temp[a] = temp[b];
                    temp[b] = t;
                }
            }
        }
        for (int z = 0; z < 6; z++) {
            sorted_draws_data[i * 6 + z] = temp[z];
        }
    }

    AnalysisResultItem* ret = (l != -1) ?
        run_standard_analysis(sorted_draws_data, draws_count - last_offset, j, k, m, l, n, max_number, out_len) :
        run_chain_analysis(sorted_draws_data, draws_count, last_offset, j, k, m, max_number, out_len);

    free(sorted_draws_data);
    return ret;
}

void free_analysis_results(AnalysisResultItem* results) {
    if (results) free(results);
}

static void format_combo(const int* combo, int len, char* out) {
    int pos = 0;
    for (int i = 0; i < len; i++) {
        if (i > 0) {
            out[pos++] = ',';
            out[pos++] = ' ';
        }
        pos += sprintf(out + pos, "%d", combo[i]);
    }
    out[pos] = '\0';
}

static void format_subsets(const int* combo, int j, int k, int total_draws,
                          const SubsetTable* table, char* out) {
    typedef struct {
        int numbers[6];
        int rank;
    } SubsetInfo;

    int exact_subset_count = (int)nCk_table[j][k];
    SubsetInfo* subsets = (SubsetInfo*)malloc(exact_subset_count * sizeof(SubsetInfo));
    if (!subsets) {
        strcpy(out, "[]");
        return;
    }
    int subset_count = 0;

    int idx[6];
    for (int i = 0; i < k; i++) idx[i] = i;
    while (1) {
        if (subset_count >= exact_subset_count) break;
        for (int i = 0; i < k; i++) {
            subsets[subset_count].numbers[i] = combo[idx[i]];
        }
        uint64 pat = numbers_to_pattern(subsets[subset_count].numbers, k);
        int last_seen = lookup_subset(table, pat);
        int rank = (last_seen >= 0) ? (total_draws - last_seen - 1) : total_draws;
        subsets[subset_count].rank = rank;
        subset_count++;
        int p = k - 1;
        while (p >= 0) {
            idx[p]++;
            if (idx[p] <= j - (k - p)) {
                for (int x = p + 1; x < k; x++) {
                    idx[x] = idx[x - 1] + 1;
                }
                break;
            }
            p--;
        }
        if (p < 0) break;
    }

    for (int i = 0; i < subset_count - 1; i++) {
        for (int j = i + 1; j < subset_count; j++) {
            if (subsets[j].rank > subsets[i].rank) {
                SubsetInfo temp = subsets[i];
                subsets[i] = subsets[j];
                subsets[j] = temp;
            }
        }
    }

    int pos = 0;
    out[pos++] = '[';
    for (int i = 0; i < subset_count; i++) {
        if (i > 0) {
            out[pos++] = ',';
            out[pos++] = ' ';
        }
        pos += sprintf(out + pos, "((%d", subsets[i].numbers[0]);
        for (int n = 1; n < k; n++) {
            pos += sprintf(out + pos, ", %d", subsets[i].numbers[n]);
        }
        pos += sprintf(out + pos, "), %d)", subsets[i].rank);
    }
    out[pos++] = ']';
    out[pos] = '\0';

    free(subsets);
}
