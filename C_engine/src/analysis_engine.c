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
#define HASH_SIZE (1 << 26) // 67M entries

typedef unsigned long long uint64;
typedef unsigned int uint32;

static uint64 nCk_table[MAX_NUMBERS][MAX_NUMBERS];
static int bit_count_table[256];
static int initialized = 0;

typedef struct {
    uint64 pattern;
    int last_seen;
} SubsetEntry;

typedef struct {
    int size;
    int capacity;
    SubsetEntry* entries;
} SubsetTable;

typedef struct {
    int combo[MAX_ALLOWED_J];
    int len;
    double avg_rank;
    double min_rank;
    uint64 pattern;
} ComboStats;

typedef struct {
    char combination[MAX_COMBO_STR];
    double avg_rank;
    double min_value;
    char subsets[MAX_SUBSETS_STR];
    int draw_offset;
    int draws_until_common;
    int analysis_start_draw;
    int is_chain_result;
} AnalysisResultItem;

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
    t->entries = (SubsetEntry*)calloc(max_entries, sizeof(SubsetEntry));
    if (!t->entries) {
        free(t);
        return NULL;
    }
    return t;
}

static void free_subset_table(SubsetTable* t) {
    if (t) {
        if (t->entries) free(t->entries);
        free(t);
    }
}

static uint64 numbers_to_pattern(const int* numbers, int count) {
    uint64 pattern = 0;
    for (int i = 0; i < count; i++) {
        pattern |= (1ULL << (numbers[i] - 1));
    }
    return pattern;
}

static int hash_pattern(uint64 pattern) {
    return (int)((pattern ^ (pattern >> 32)) & (HASH_SIZE - 1));
}

static void insert_subset(SubsetTable* table, uint64 pattern, int last_seen) {
    if (table->size >= table->capacity) return;

    int idx = hash_pattern(pattern);
    while (table->entries[idx].pattern != 0 && table->entries[idx].pattern != pattern) {
        idx = (idx + 1) & (HASH_SIZE - 1);
    }

    table->entries[idx].pattern = pattern;
    table->entries[idx].last_seen = last_seen;
    table->size++;
}

static int lookup_subset(SubsetTable* table, uint64 pattern) {
    int idx = hash_pattern(pattern);
    while (table->entries[idx].pattern != 0) {
        if (table->entries[idx].pattern == pattern) {
            return table->entries[idx].last_seen;
        }
        idx = (idx + 1) & (HASH_SIZE - 1);
    }
    return -1;
}

static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table) {
    int idx[MAX_ALLOWED_J];
    for (int i = 0; i < k; i++) {
        idx[i] = i;
    }

    int subset_count = 0;
    int max_num = 0;
    for (int i = 0; draw[i] != 0; i++) {
        max_num = draw[i];
    }

    while (1) {
        int subset[MAX_ALLOWED_J];
        for (int i = 0; i < k; i++) {
            subset[i] = draw[idx[i]];
        }

        uint64 pat = numbers_to_pattern(subset, k);
        insert_subset(table, pat, draw_idx);

        subset_count++;
        int p = k - 1;
        while (p >= 0 && idx[p] == max_num - k + p) p--;
        if (p < 0) break;

        idx[p]++;
        for (int x = p + 1; x < k; x++) {
            idx[x] = idx[x-1] + 1;
        }
    }
}

static void evaluate_combo(const int* combo, int j, int k, int total_draws, const SubsetTable* table, ComboStats* stats) {
    int idx[MAX_ALLOWED_J];
    for (int i = 0; i < k; i++) {
        idx[i] = i;
    }

    double sum_ranks = 0;
    double min_rank = total_draws;
    int count = 0;

    while (1) {
        int subset[MAX_ALLOWED_J];
        for (int i = 0; i < k; i++) {
            subset[i] = combo[idx[i]];
        }

        uint64 pat = numbers_to_pattern(subset, k);
        int last_seen = lookup_subset(table, pat);
        double rank = (last_seen >= 0) ? (double)(total_draws - last_seen - 1) : (double)total_draws;

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
            idx[x] = idx[x-1] + 1;
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
        }
        pos += sprintf(out + pos, "%d", combo[i]);
    }
    out[pos] = '\0';
}

static void format_subsets(const int* combo, int j, int k, int total_draws, const SubsetTable* table, char* out) {
    int idx[MAX_ALLOWED_J];
    for (int i = 0; i < k; i++) {
        idx[i] = i;
    }

    SubsetInfo* subsets = (SubsetInfo*)malloc(nCk_table[j][k] * sizeof(SubsetInfo));
    if (!subsets) return;

    int subset_count = 0;
    while (1) {
        for (int i = 0; i < k; i++) {
            subsets[subset_count].numbers[i] = combo[idx[i]];
        }

        uint64 pat = numbers_to_pattern(subsets[subset_count].numbers, k);
        int last_seen = lookup_subset(table, pat);
        subsets[subset_count].rank = (last_seen >= 0) ? (total_draws - last_seen - 1) : total_draws;
        subset_count++;

        int p = k - 1;
        while (p >= 0 && idx[p] == j - k + p) p--;
        if (p < 0) break;

        idx[p]++;
        for (int x = p + 1; x < k; x++) {
            idx[x] = idx[x-1] + 1;
        }
    }

    // Sort by rank (descending)
    for (int i = 0; i < subset_count - 1; i++) {
        for (int j = i + 1; j < subset_count; j++) {
            if (subsets[j].rank > subsets[i].rank) {
                SubsetInfo temp = subsets[i];
                subsets[i] = subsets[j];
                subsets[j] = temp;
            }
        }
    }

    int remaining_space = MAX_SUBSETS_STR;
    int pos = 0;
    out[pos++] = '[';
    remaining_space--;

    for (int i = 0; i < subset_count && remaining_space > 0; i++) {
        if (i > 0) {
            if (remaining_space < 2) break;
            out[pos++] = ',';
            out[pos++] = ' ';
            remaining_space -= 2;
        }

        int subset_len = sprintf(out + pos, "((%d", subsets[i].numbers[0]);
        pos += subset_len;
        remaining_space -= subset_len;

        for (int n = 1; n < k; n++) {
            if (remaining_space < 4) break;
            subset_len = sprintf(out + pos, ", %d", subsets[i].numbers[n]);
            pos += subset_len;
            remaining_space -= subset_len;
        }

        if (remaining_space < 4) break;
        subset_len = sprintf(out + pos, "), %d)", subsets[i].rank);
        pos += subset_len;
        remaining_space -= subset_len;
    }

    if (remaining_space > 0) {
        out[pos++] = ']';
    } else {
        out[MAX_SUBSETS_STR-1] = ']';
        pos = MAX_SUBSETS_STR;
    }
    out[pos] = '\0';

    free(subsets);
}

static AnalysisResultItem* run_standard_analysis(const int* sorted_draws_data, int use_count, int j, int k, const char* m, int l, int n, int max_number, const SubsetTable* table, int* out_len) {
    // 1) Prepare shared data structures
    ComboStats* best_stats = (ComboStats*)malloc(l * sizeof(ComboStats));
    if (!best_stats) return NULL;
    memset(best_stats, 0, l * sizeof(ComboStats));

    AnalysisResultItem* results = (AnalysisResultItem*)calloc(l + n, sizeof(AnalysisResultItem));
    if (!results) {
        free(best_stats);
        return NULL;
    }

    int filled = 0;
    int error_occurred = 0;

    // 2) Prepare thread-local data structures
    #pragma omp parallel
    {
        int* curr_combo = NULL;
        ComboStats* thread_best = NULL;
        int thread_filled = 0;

        curr_combo = (int*)malloc(j * sizeof(int));
        thread_best = (ComboStats*)malloc(l * sizeof(ComboStats));

        if (!curr_combo || !thread_best) {
            #pragma omp atomic write
            error_occurred = 1;
        }

        if (!error_occurred) {
            memset(thread_best, 0, l * sizeof(ComboStats));

            // Initialize combination with first j numbers
            for (int i = 0; i < j; i++) {
                curr_combo[i] = i + 1;
            }

            uint64 Cjk = nCk_table[j][k];

            while (1) {
                // Skip invalid combinations (numbers > max_number)
                int valid = 1;
                for (int i = 0; i < j; i++) {
                    if (curr_combo[i] > max_number) {
                        valid = 0;
                        break;
                    }
                }
                if (!valid) break;

                ComboStats stats;
                evaluate_combo(curr_combo, j, k, use_count, table, &stats);

                // Update thread_best array
                if (thread_filled < l) {
                    memcpy(&thread_best[thread_filled], &stats, sizeof(ComboStats));
                    thread_filled++;

                    // Bubble up in thread_best
                    for (int i = thread_filled - 1; i > 0; i--) {
                        int swap = 0;
                        if (strcmp(m, "avg") == 0) {
                            swap = (thread_best[i].avg_rank > thread_best[i-1].avg_rank) ||
                                   (thread_best[i].avg_rank == thread_best[i-1].avg_rank &&
                                    thread_best[i].min_rank > thread_best[i-1].min_rank);
                        } else {
                            swap = (thread_best[i].min_rank > thread_best[i-1].min_rank) ||
                                   (thread_best[i].min_rank == thread_best[i-1].min_rank &&
                                    thread_best[i].avg_rank > thread_best[i-1].avg_rank);
                        }

                        if (swap) {
                            ComboStats tmp = thread_best[i];
                            thread_best[i] = thread_best[i-1];
                            thread_best[i-1] = tmp;
                        } else {
                            break;
                        }
                    }
                } else {
                    int should_replace = 0;
                    if (strcmp(m, "avg") == 0) {
                        should_replace = (stats.avg_rank > thread_best[l-1].avg_rank) ||
                                        (stats.avg_rank == thread_best[l-1].avg_rank &&
                                         stats.min_rank > thread_best[l-1].min_rank);
                    } else {
                        should_replace = (stats.min_rank > thread_best[l-1].min_rank) ||
                                        (stats.min_rank == thread_best[l-1].min_rank &&
                                         stats.avg_rank > thread_best[l-1].avg_rank);
                    }

                    if (should_replace) {
                        thread_best[l-1] = stats;

                        // Bubble up
                        for (int i = l-1; i > 0; i--) {
                            int should_bubble = 0;
                            if (strcmp(m, "avg") == 0) {
                                should_bubble = (thread_best[i].avg_rank > thread_best[i-1].avg_rank) ||
                                               (thread_best[i].avg_rank == thread_best[i-1].avg_rank &&
                                                thread_best[i].min_rank > thread_best[i-1].min_rank);
                            } else {
                                should_bubble = (thread_best[i].min_rank > thread_best[i-1].min_rank) ||
                                               (thread_best[i].min_rank == thread_best[i-1].min_rank &&
                                                thread_best[i].avg_rank > thread_best[i-1].avg_rank);
                            }

                            if (should_bubble) {
                                ComboStats tmp = thread_best[i];
                                thread_best[i] = thread_best[i-1];
                                thread_best[i-1] = tmp;
                            } else {
                                break;
                            }
                        }
                    }
                }

                // Generate next combination
                int pos = j - 1;
                while (pos >= 0 && curr_combo[pos] == max_number - j + pos + 1) pos--;
                if (pos < 0) break;

                curr_combo[pos]++;
                for (int x = pos + 1; x < j; x++) {
                    curr_combo[x] = curr_combo[pos] + (x - pos);
                }
            }
        }

        // Merge thread results into global best_stats
        #pragma omp critical
        {
            for (int i = 0; i < thread_filled; i++) {
                if (filled < l) {
                    memcpy(&best_stats[filled], &thread_best[i], sizeof(ComboStats));
                    filled++;
                } else {
                    double val = (strcmp(m, "avg") == 0) ? thread_best[i].avg_rank : thread_best[i].min_rank;
                    double worst_val = (strcmp(m, "avg") == 0) ? best_stats[l-1].avg_rank : best_stats[l-1].min_rank;

                    if (val > worst_val) {
                        best_stats[l-1] = thread_best[i];

                        // Bubble up
                        for (int idx = l-1; idx > 0; idx--) {
                            double vcur = (strcmp(m, "avg") == 0) ? best_stats[idx].avg_rank : best_stats[idx].min_rank;
                            double vprev = (strcmp(m, "avg") == 0) ? best_stats[idx-1].avg_rank : best_stats[idx-1].min_rank;

                            if (vcur > vprev) {
                                ComboStats tmp = best_stats[idx];
                                best_stats[idx] = best_stats[idx-1];
                                best_stats[idx-1] = tmp;
                            } else {
                                break;
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
        return NULL;
    }

    // Sort best_stats by the appropriate metric
    for (int i = 0; i < filled - 1; i++) {
        for (int j = i + 1; j < filled; j++) {
            int swap = 0;
            if (strcmp(m, "avg") == 0) {
                swap = (best_stats[j].avg_rank > best_stats[i].avg_rank) ||
                       (best_stats[j].avg_rank == best_stats[i].avg_rank &&
                        best_stats[j].min_rank > best_stats[i].min_rank);
            } else {
                swap = (best_stats[j].min_rank > best_stats[i].min_rank) ||
                       (best_stats[j].min_rank == best_stats[i].min_rank &&
                        best_stats[j].avg_rank > best_stats[i].avg_rank);
            }

            if (swap) {
                ComboStats tmp = best_stats[i];
                best_stats[i] = best_stats[j];
                best_stats[j] = tmp;
            }
        }
    }

    // Fill results array
    int results_count = 0;
    for (int i = 0; i < filled && results_count < l; i++) {
        format_combo(best_stats[i].combo, best_stats[i].len, results[results_count].combination);
        format_subsets(best_stats[i].combo, j, k, use_count, table, results[results_count].subsets);
        results[results_count].avg_rank = best_stats[i].avg_rank;
        results[results_count].min_value = best_stats[i].min_rank;
        results[results_count].is_chain_result = 0;
        results_count++;
    }

    // Handle non-overlapping selections if n > 0
    int second_table_count = 0;
    int* pick_indices = NULL;

    if (n > 0 && results_count > 0) {
        pick_indices = (int*)malloc(results_count * sizeof(int));
        memset(pick_indices, -1, results_count * sizeof(int));

        int chosen = 0;
        pick_indices[chosen++] = 0;

        for (int i = 1; i < results_count && chosen < n; i++) {
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

    if (pick_indices) free(pick_indices);
    free(best_stats);

    *out_len = results_count + second_table_count;
    return results;
}

static AnalysisResultItem* run_chain_analysis(const int* sorted_draws_data, int use_count, int j, int k, int l, int n, int max_number, const SubsetTable* table, int* out_len) {
    // Implementation for chain analysis
    // This is a placeholder - the actual implementation would depend on requirements
    *out_len = 0;
    return NULL;
}

AnalysisResultItem* run_analysis_c(const char* game_type, int** draws, int draws_count, int j, int k, const char* m, int l, int n, int last_offset, int* out_len) {
    if (j < k || j > MAX_ALLOWED_J || k < 1 || l < 1 || draws_count == 0) {
        *out_len = 0;
        return NULL;
    }

    init_tables();

    // Determine max_number based on game_type
    int max_number = 49; // default
    if (strcmp(game_type, "6_42") == 0) max_number = 42;
    else if (strcmp(game_type, "6_49") == 0) max_number = 49;

    // Use all draws if last_offset is 0, otherwise use a subset
    int use_count = (last_offset > 0) ? last_offset : draws_count;
    if (use_count > draws_count) use_count = draws_count;

    // Create and populate subset table
    SubsetTable* table = create_subset_table(HASH_SIZE);
    if (!table) {
        *out_len = 0;
        return NULL;
    }

    // Process each draw to extract k-subsets
    for (int i = 0; i < use_count; i++) {
        process_draw(draws[i], i, k, table);
    }

    AnalysisResultItem* results = NULL;

    // Run appropriate analysis
    if (l != -1) {
        results = run_standard_analysis(NULL, use_count, j, k, m, l, n, max_number, table, out_len);
    } else {
        results = run_chain_analysis(NULL, use_count, j, k, l, n, max_number, table, out_len);
    }

    free_subset_table(table);
    return results;
}
