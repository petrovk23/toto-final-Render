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
 * ComboStats holds the enumerated combo stats for the top-l searching.
 * 'pattern' is the bit pattern (1<<(num-1)) of the combo's numbers.
 */
typedef struct {
    uint64 pattern;   // bit pattern of combo
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
static void evaluate_combo(const int* combo, int j, int k, int total_draws,
                           const SubsetTable* table, ComboStats* stats);
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

/**
 * run_analysis_c(...)
 * -------------------
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
) {
    *out_len = 0;
    if (j > MAX_ALLOWED_J) {
        return NULL;
    }
    init_tables();

    // Decide max_number from the game_type
    int max_number = (strstr(game_type, "6_49")) ? 49 : 42;
    if (draws_count < 1) {
        return NULL;
    }

    // Build a local "sorted_draws_data" array. Each draw is sorted ascending.
    // The DB supplies them oldest->newest in index 0..draws_count-1.
    // We'll keep that order: index 0 => oldest, index draws_count-1 => newest.
    int* sorted_draws_data = (int*)malloc(draws_count * 6 * sizeof(int));
    if (!sorted_draws_data) {
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        int temp[6];
        for (int z = 0; z < 6; z++) {
            temp[z] = draws[i][z];
        }
        // sort the 6 numbers
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

    // If l != -1, run the standard approach
    if (l != -1) {
        int use_count = draws_count - last_offset;
        if (use_count < 1) {
            free(sorted_draws_data);
            return NULL;
        }
        AnalysisResultItem* ret = run_standard_analysis(
            sorted_draws_data,
            use_count,
            j, k, m, l, n, max_number,
            out_len
        );
        free(sorted_draws_data);
        return ret;
    }

    // Otherwise, chain analysis
    AnalysisResultItem* chain_ret = run_chain_analysis(
        sorted_draws_data,
        draws_count,
        last_offset,
        j, k, m,
        max_number,
        out_len
    );
    free(sorted_draws_data);
    return chain_ret;
}

// ----------------------------------------------------------------------
// Standard top-l analysis (and up to n combos that do not overlap in k-subsets).
// ----------------------------------------------------------------------
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
    // 1) Build the subset table from the newest use_count draws
    SubsetTable* table = create_subset_table(HASH_SIZE);
    if (!table) return NULL;
    for (int i = 0; i < use_count; i++) {
        process_draw(&sorted_draws_data[i * 6], i, k, table);
    }

    // 2) Enumerate all j-combinations, find top-l. We'll store them in best_stats.
    // 2) Prepare shared and thread-local data structures
    ComboStats* best_stats = (ComboStats*)malloc(l * sizeof(ComboStats));
    if (!best_stats) {
        free_subset_table(table);
        return NULL;
    }
    memset(best_stats, 0, l * sizeof(ComboStats));

    AnalysisResultItem* results = (AnalysisResultItem*)calloc(l + n, sizeof(AnalysisResultItem));
    if (!results) {
        free(best_stats);
        free_subset_table(table);
        return NULL;
    }

    int filled = 0;  // shared counter for best_stats filling
    int error_occurred = 0;  // flag for allocation errors

    #pragma omp parallel
    {
        // Thread-local combo buffer
        int* curr_combo = NULL;
        ComboStats* thread_best = NULL;
        int thread_filled = 0;

        curr_combo = (int*)malloc(j * sizeof(int));
        if (!curr_combo) {
            #pragma omp atomic write
            error_occurred = 1;
        } else {
            thread_best = (ComboStats*)malloc(l * sizeof(ComboStats));
            if (!thread_best) {
                free(curr_combo);
                #pragma omp atomic write
                error_occurred = 1;
            } else {
                memset(thread_best, 0, l * sizeof(ComboStats));

                // Parallelize by first number
                #pragma omp for schedule(dynamic)
                for (int first = 1; first <= max_number - j + 1; first++) {
                    if (!error_occurred) {  // Check if any thread had allocation error
                        curr_combo[0] = first;
                        for (int i = 1; i < j; i++) {
                            curr_combo[i] = first + i;
                        }

                        while (1) {
                            ComboStats stats;
                            evaluate_combo(curr_combo, j, k, use_count, table, &stats);

                            // Update thread_best array
                            if (thread_filled < l) {
                                memcpy(&thread_best[thread_filled], &stats, sizeof(ComboStats));
                                thread_filled++;
                                // bubble up in thread_best
                                for (int i = thread_filled - 1; i > 0; i--) {
                                    int swap;
                                    if (strcmp(m, "avg") == 0) {
                                        // For 'avg': first by avg_rank, then by min_rank
                                        swap = (thread_best[i].avg_rank > thread_best[i - 1].avg_rank) ||
                                            (thread_best[i].avg_rank == thread_best[i - 1].avg_rank &&
                                                thread_best[i].min_rank > thread_best[i - 1].min_rank);
                                    } else {
                                        // For 'min': first by min_rank, then by avg_rank
                                        swap = (thread_best[i].min_rank > thread_best[i - 1].min_rank) ||
                                            (thread_best[i].min_rank == thread_best[i - 1].min_rank &&
                                                thread_best[i].avg_rank > thread_best[i - 1].avg_rank);
                                    }
                                    if (swap) {
                                        ComboStats tmp = thread_best[i];
                                        thread_best[i] = thread_best[i - 1];
                                        thread_best[i - 1] = tmp;
                                    } else {
                                        break;
                                    }
                                }
                            } else {
                                int should_replace = 0;
                                if (strcmp(m, "avg") == 0) {
                                    // For 'avg': first by avg_rank, then by min_rank
                                    should_replace = (stats.avg_rank > thread_best[l - 1].avg_rank) ||
                                                    (stats.avg_rank == thread_best[l - 1].avg_rank &&
                                                    stats.min_rank > thread_best[l - 1].min_rank);
                                } else {
                                    // For 'min': first by min_rank, then by avg_rank
                                    should_replace = (stats.min_rank > thread_best[l - 1].min_rank) ||
                                                    (stats.min_rank == thread_best[l - 1].min_rank &&
                                                    stats.avg_rank > thread_best[l - 1].avg_rank);
                                }
                                if (should_replace) {
                                    thread_best[l - 1] = stats;
                                    // bubble up
                                    for (int i = l - 1; i > 0; i--) {
                                        int should_bubble = 0;
                                        if (strcmp(m, "avg") == 0) {
                                            // For 'avg': first by avg_rank, then by min_rank
                                            should_bubble = (thread_best[i].avg_rank > thread_best[i - 1].avg_rank) ||
                                                        (thread_best[i].avg_rank == thread_best[i - 1].avg_rank &&
                                                        thread_best[i].min_rank > thread_best[i - 1].min_rank);
                                        } else {
                                            // For 'min': first by min_rank, then by avg_rank
                                            should_bubble = (thread_best[i].min_rank > thread_best[i - 1].min_rank) ||
                                                        (thread_best[i].min_rank == thread_best[i - 1].min_rank &&
                                                        thread_best[i].avg_rank > thread_best[i - 1].avg_rank);
                                        }
                                        if (should_bubble) {
                                            ComboStats tmp = thread_best[i];
                                            thread_best[i] = thread_best[i - 1];
                                            thread_best[i - 1] = tmp;
                                        } else {
                                            break;
                                        }
                                    }
                                }
                            }

                            // Next combo
                            int pos = j - 1;
                            while (pos >= 0 && curr_combo[pos] == max_number - j + pos + 1) pos--;
                            if (pos < 0 || pos == 0) break;  // Break if done or first number would change
                            curr_combo[pos]++;
                            for (int x = pos + 1; x < j; x++) {
                                curr_combo[x] = curr_combo[pos] + (x - pos);
                            }
                        }
                    }
                }

                // Merge thread results into global best_stats
                #pragma omp critical
                {
                    // Merge thread_best into best_stats
                    for (int i = 0; i < thread_filled; i++) {
                        if (filled < l) {
                            memcpy(&best_stats[filled], &thread_best[i], sizeof(ComboStats));
                            filled++;
                        } else {
                            double val = (strcmp(m, "avg") == 0)
                                ? thread_best[i].avg_rank
                                : thread_best[i].min_rank;
                            double worst_val = (strcmp(m, "avg") == 0)
                                ? best_stats[l - 1].avg_rank
                                : best_stats[l - 1].min_rank;
                            if (val > worst_val) {
                                best_stats[l - 1] = thread_best[i];
                                // bubble up
                                for (int idx = l - 1; idx > 0; idx--) {
                                    double vcur = (strcmp(m, "avg") == 0)
                                        ? best_stats[idx].avg_rank
                                        : best_stats[idx].min_rank;
                                    double vprev = (strcmp(m, "avg") == 0)
                                        ? best_stats[idx - 1].avg_rank
                                        : best_stats[idx - 1].min_rank;
                                    if (vcur > vprev) {
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
        }

        if (thread_best) free(thread_best);
        if (curr_combo) free(curr_combo);
    }

    if (error_occurred) {
        free(best_stats);
        free(results);
        free_subset_table(table);
        return NULL;
    }
    // 3) Now fill the top-l combos into results
    int top_count = (filled < l) ? filled : l;  // how many combos we actually have
    int results_count = 0;

    // Rebuild table once more just to format subsets easily
    // (Slight overhead, but simpler.)
    free_subset_table(table);
    table = create_subset_table(HASH_SIZE);
    for (int i = 0; i < use_count; i++) {
        process_draw(&sorted_draws_data[i * 6], i, k, table);
    }

    for (int i = 0; i < top_count; i++) {
        // fill results[i] with the i-th best combo
        format_combo(best_stats[i].combo, best_stats[i].len, results[results_count].combination);
        format_subsets(best_stats[i].combo, j, k, use_count, table, results[results_count].subsets);
        results[results_count].avg_rank = best_stats[i].avg_rank;
        results[results_count].min_value = best_stats[i].min_rank;
        results[results_count].is_chain_result = 0;
        results[results_count].draw_offset = 0;
        results[results_count].analysis_start_draw = 0;
        results[results_count].draws_until_common = 0;
        // Also store the bit pattern in the struct so we can check overlap later
        // We'll repurpose analysis_start_draw to store the pattern. But better is just to store it in an array:
        // Actually we'll store it in best_stats[i].pattern. Already there.
        results_count++;
    }

    // 4) If n > 0, find up to n combos from these top_count combos
    //    that do not share a k-subset among themselves. We do this
    //    by scanning in the same order as top-l, skipping overlaps.
    int second_table_count = 0;
    int* pick_indices = NULL;
    if (n > 0 && top_count > 0) {
        pick_indices = (int*)malloc(top_count * sizeof(int));
        memset(pick_indices, -1, top_count * sizeof(int));

        // The first chosen combo is always best_stats[0].
        int chosen = 0;
        pick_indices[chosen++] = 0;

        for (int i = 1; i < top_count && chosen < n; i++) {
            // check overlap with all chosen combos
            uint64 pat_i = best_stats[i].pattern;
            int overlap = 0;
            for (int c = 0; c < chosen; c++) {
                int idxC = pick_indices[c];
                uint64 pat_c = best_stats[idxC].pattern;
                // Overlap if popcount64(pat_i & pat_c) >= k
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
        second_table_count = chosen; // how many we picked
    }

    // Now fill those second-table combos in the same results array
    // right after the top-l combos
    // Each one gets subsets too, so we do the same table logic
    int bottom_start = results_count;
    for (int i = 0; i < second_table_count; i++) {
        int idx = pick_indices[i];
        format_combo(best_stats[idx].combo, best_stats[idx].len, results[bottom_start + i].combination);
        format_subsets(best_stats[idx].combo, j, k, use_count, table, results[bottom_start + i].subsets);
        results[bottom_start + i].avg_rank = best_stats[idx].avg_rank;
        results[bottom_start + i].min_value = best_stats[idx].min_rank;
        results[bottom_start + i].is_chain_result = 0;
        results[bottom_start + i].draw_offset = 0;
        results[bottom_start + i].analysis_start_draw = 0;
        results[bottom_start + i].draws_until_common = 0;
    }
    int total_used = results_count + second_table_count;
    *out_len = total_used;

    if (pick_indices) {
        free(pick_indices);
    }
    free_subset_table(table);
    free(best_stats);

    if (total_used == 0) {
        free(results);
        return NULL;
    }
    return results;
}

// ----------------------------------------------------------------------
// Chain analysis (l = -1).
// ----------------------------------------------------------------------
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
    // We'll store results in a dynamic array of size (initial_offset+2).
    // Each chain iteration yields 1 combo.
    AnalysisResultItem* chain_results = (AnalysisResultItem*)calloc(initial_offset + 2, sizeof(AnalysisResultItem));
    if (!chain_results) {
        *out_len = 0;
        return NULL;
    }

    // Precompute bit patterns for each draw (6 numbers => bit pattern).
    uint64* draw_patterns = (uint64*)malloc(draws_count * sizeof(uint64));
    if (!draw_patterns) {
        free(chain_results);
        *out_len = 0;
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        uint64 pat = 0ULL;
        for (int z = 0; z < 6; z++) {
            pat |= (1ULL << (sorted_draws_data[i * 6 + z] - 1));
        }
        draw_patterns[i] = pat;
    }

    int chain_index = 0;
    int current_offset = initial_offset;

    while (1) {
        if (current_offset < 0) {
            break;
        }
        if (current_offset > (draws_count - 1)) {
            // no valid draws to analyze
            break;
        }
        int use_count = draws_count - current_offset;
        if (use_count < 1) {
            break;
        }

        // Build subset table for newest use_count draws
        SubsetTable* table = create_subset_table(HASH_SIZE);
        for (int i = 0; i < use_count; i++) {
            process_draw(&sorted_draws_data[i * 6], i, k, table);
        }

        // We want top-1 combo under j,k,m
        int found_any = 0;
        double best_val = -1e9;
        ComboStats best_stat;
        memset(&best_stat, 0, sizeof(best_stat));

        // Enumerate
        int* combo_buf = (int*)malloc(j * sizeof(int));
        if (!combo_buf) {
            free_subset_table(table);
            break;
        }
        for (int i = 0; i < j; i++) {
            combo_buf[i] = i + 1;
        }

        while (1) {
            ComboStats stats;
            evaluate_combo(combo_buf, j, k, use_count, table, &stats);

            int is_better = 0;
            if (!found_any) {
                is_better = 1;
            } else if (strcmp(m, "avg") == 0) {
                // For 'avg': first by avg_rank, then by min_rank
                is_better = (stats.avg_rank > best_stat.avg_rank) ||
                            (stats.avg_rank == best_stat.avg_rank &&
                            stats.min_rank > best_stat.min_rank);
            } else {
                // For 'min': first by min_rank, then by avg_rank
                is_better = (stats.min_rank > best_stat.min_rank) ||
                            (stats.min_rank == best_stat.min_rank &&
                            stats.avg_rank > best_stat.avg_rank);
            }
            if (is_better) {
                best_stat = stats;
                found_any = 1;
            }

            // next
            int pos = j - 1;
            while (pos >= 0 && combo_buf[pos] == max_number - j + pos + 1) pos--;
            if (pos < 0) break;
            combo_buf[pos]++;
            for (int x = pos + 1; x < j; x++) {
                combo_buf[x] = combo_buf[pos] + (x - pos);
            }
        }
        free(combo_buf);
        free_subset_table(table);

        if (!found_any) {
            break;
        }

        // Fill chain_results item
        AnalysisResultItem* out_item = &chain_results[chain_index];
        format_combo(best_stat.combo, best_stat.len, out_item->combination);

        // Build a subsets string for display
        {
            SubsetTable* tmp_t = create_subset_table(HASH_SIZE);
            for (int i = 0; i < use_count; i++) {
                process_draw(&sorted_draws_data[i * 6], i, k, tmp_t);
            }
            format_subsets(best_stat.combo, j, k, use_count, tmp_t, out_item->subsets);
            free_subset_table(tmp_t);
        }
        out_item->avg_rank = best_stat.avg_rank;
        out_item->min_value = best_stat.min_rank;
        out_item->is_chain_result = 1;
        out_item->draw_offset = chain_index + 1;  // "Analysis #"
        out_item->analysis_start_draw = draws_count - current_offset;  // "For Draw"

        // Now find forward draws that share a k-subset
        // with the best_stat combo
        uint64 combo_pat = 0ULL;
        for (int z = 0; z < j; z++) {
            combo_pat |= (1ULL << (best_stat.combo[z] - 1));
        }

        int found_common = 0;
        int i;
        for (i = 1; i <= current_offset; i++) {
            int f_idx = draws_count - 1 - (current_offset - i);
            if (f_idx < 0) break;
            uint64 fpat = draw_patterns[f_idx];
            uint64 inter = (combo_pat & fpat);
            if (popcount64(inter) >= k) {
                found_common = 1;
                break;
            }
        }
        // If not found, pretend we found it after current_offset + 1
        if (!found_common) {
            i = current_offset + 1;
        } else if (i > current_offset) {
            i = current_offset;
        }

        out_item->draws_until_common = (i > 0) ? (i - 1) : 0;
        current_offset -= i;
        chain_index++;

        if (current_offset <= 0) {
            break;
        }
    }

    free(draw_patterns);
    *out_len = chain_index;
    if (chain_index == 0) {
        free(chain_results);
        return NULL;
    }
    return chain_results;
}

/**
 * free_analysis_results(...)
 */
void free_analysis_results(AnalysisResultItem* results) {
    if (results) {
        free(results);
    }
}

// ----------------------------------------------------------------------
// Implementation details
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
    // bit counts
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
    int res = 0;
    for (int i = 0; i < 8; i++) {
        res += bit_count_table[x & 0xFF];
        x >>= 8;
    }
    return res;
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
    // Simple 64->32 hash
    pattern ^= pattern >> 33;
    pattern *= 0xff51afd7ed558ccdULL;
    pattern ^= pattern >> 33;
    pattern *= 0xc4ceb9fe1a85ec53ULL;
    pattern ^= pattern >> 33;
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
    int idx[20];
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
        while (pos >= 0 && idx[pos] == 6 - k + pos) pos--;
        if (pos < 0) break;
        idx[pos]++;
        for (int x = pos + 1; x < k; x++) {
            idx[x] = idx[x - 1] + 1;
        }
    }
}

static void evaluate_combo(const int* combo, int j, int k, int total_draws,
                           const SubsetTable* table, ComboStats* stats)
{
    double sum_ranks = 0.0;
    double min_rank = (double)total_draws;
    int count = 0;

    int idx[20];
    for (int i = 0; i < k; i++) {
        idx[i] = i;
    }

    while (1) {
        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) {
            pat |= (1ULL << (combo[idx[i]] - 1));
        }
        int last_seen = lookup_subset(table, pat);
        double rank = (last_seen >= 0)
                      ? (double)(total_draws - last_seen - 1)
                      : (double)total_draws;
        sum_ranks += rank;
        if (rank < min_rank) {
            min_rank = rank;
        }
        count++;

        int pos = k - 1;
        while (pos >= 0 && idx[pos] == j - k + pos) pos--;
        if (pos < 0) break;
        idx[pos]++;
        for (int x = pos + 1; x < k; x++) {
            idx[x] = idx[x - 1] + 1;
        }
    }

    stats->pattern = numbers_to_pattern(combo, j);
    stats->avg_rank = sum_ranks / (double)count;
    stats->min_rank = min_rank;
    memcpy(stats->combo, combo, j * sizeof(int));
    stats->len = j;
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
                           const SubsetTable* table, char* out)
{
    int pos = 0;
    out[pos++] = '[';

    int idx[20];
    for (int i = 0; i < k; i++) {
        idx[i] = i;
    }
    int first = 1;

    while (1) {
        if (!first && pos < (MAX_SUBSETS_STR - 2)) {
            out[pos++] = ',';
            out[pos++] = ' ';
        }
        first = 0;
        if (pos >= MAX_SUBSETS_STR - 20) break;

        out[pos++] = '(';
        out[pos++] = '(';

        for (int i = 0; i < k; i++) {
            if (i > 0) {
                out[pos++] = ',';
                out[pos++] = ' ';  // Add space after comma
            }
            pos += sprintf(out + pos, "%d", combo[idx[i]]);
            if (pos >= MAX_SUBSETS_STR - 10) break;
        }
        out[pos++] = ')';
        out[pos++] = ',';
        out[pos++] = ' ';

        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) {
            pat |= (1ULL << (combo[idx[i]] - 1));
        }
        int last_seen = lookup_subset(table, pat);
        int rank = (last_seen >= 0)
                   ? (total_draws - last_seen - 1)
                   : total_draws;
        pos += sprintf(out + pos, "%d)", rank);
        if (pos >= MAX_SUBSETS_STR - 5) break;

        // next k-subset
        int p = k - 1;
        while (p >= 0 && idx[p] == j - k + p) p--;
        if (p < 0) break;
        idx[p]++;
        for (int x = p + 1; x < k; x++) {
            idx[x] = idx[x - 1] + 1;
        }
    }

    if (pos < MAX_SUBSETS_STR) {
        out[pos++] = ']';
    }
    out[pos] = '\0';
}
