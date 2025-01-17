#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "analysis_engine.h"

#define MAX_COMBO_STR 255
#define MAX_SUBSETS_STR 511
#define MAX_ALLOWED_J 200
#define MAX_ALLOWED_OUT_LEN 1000000
#define MAX_NUMBERS 50
#define HASH_SIZE (1 << 20)  // 1M entries

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
static inline int popcount(uint64 x);
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

// Two helper routines
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
// run_analysis_c(...)
// ----------------------------------------------------------------------
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
            sorted_draws_data[i*6 + z] = temp[z];
        }
    }

    // If l != -1, we do the standard top-l approach
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
// Standard top-l (and optional +n) analysis (non-chain)
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
    SubsetTable* table = create_subset_table(HASH_SIZE);
    if (!table) {
        return NULL;
    }

    // Build subset table from the newest use_count draws (indexes 0..use_count-1)
    // so the last of those is index use_count-1 => newest
    for (int i = 0; i < use_count; i++) {
        process_draw(&sorted_draws_data[i*6], i, k, table);
    }

    // We'll accumulate up to (l + n) results
    AnalysisResultItem* results = (AnalysisResultItem*)calloc(l + n, sizeof(AnalysisResultItem));
    if (!results) {
        free_subset_table(table);
        return NULL;
    }

    // We'll store the top-l combos in a small array
    ComboStats* best_stats = (ComboStats*)malloc(l * sizeof(ComboStats));
    if (!best_stats) {
        free(results);
        free_subset_table(table);
        return NULL;
    }
    memset(best_stats, 0, l * sizeof(ComboStats));

    int* curr_combo = (int*)malloc(j * sizeof(int));
    if (!curr_combo) {
        free(best_stats);
        free(results);
        free_subset_table(table);
        return NULL;
    }

    // init [1..j]
    for (int i = 0; i < j; i++) {
        curr_combo[i] = i + 1;
    }

    int filled = 0;
    while (1) {
        // evaluate
        ComboStats stats;
        evaluate_combo(curr_combo, j, k, use_count, table, &stats);

        if (filled < l) {
            memcpy(&best_stats[filled], &stats, sizeof(ComboStats));
            filled++;
            // bubble up
            for (int i = filled - 1; i > 0; i--) {
                int swap;
                if (strcmp(m, "avg") == 0) {
                    swap = (best_stats[i].avg_rank > best_stats[i-1].avg_rank);
                } else {
                    swap = (best_stats[i].min_rank > best_stats[i-1].min_rank);
                }
                if (swap) {
                    ComboStats temp = best_stats[i];
                    best_stats[i] = best_stats[i-1];
                    best_stats[i-1] = temp;
                } else {
                    break;
                }
            }
        } else {
            // compare vs best_stats[l-1]
            int replace;
            if (strcmp(m, "avg") == 0) {
                replace = (stats.avg_rank > best_stats[l-1].avg_rank);
            } else {
                replace = (stats.min_rank > best_stats[l-1].min_rank);
            }
            if (replace) {
                best_stats[l-1] = stats;
                // bubble up
                for (int i = l - 1; i > 0; i--) {
                    int swap;
                    if (strcmp(m, "avg") == 0) {
                        swap = (best_stats[i].avg_rank > best_stats[i-1].avg_rank);
                    } else {
                        swap = (best_stats[i].min_rank > best_stats[i-1].min_rank);
                    }
                    if (swap) {
                        ComboStats temp = best_stats[i];
                        best_stats[i] = best_stats[i-1];
                        best_stats[i-1] = temp;
                    } else {
                        break;
                    }
                }
            }
        }

        // next j-combo
        int pos = j - 1;
        while (pos >= 0 && curr_combo[pos] == max_number - j + pos + 1) pos--;
        if (pos < 0) break;
        curr_combo[pos]++;
        for (int x = pos + 1; x < j; x++) {
            curr_combo[x] = curr_combo[pos] + (x - pos);
        }
    }

    free(curr_combo);
    free_subset_table(table);

    // fill results
    int results_count = 0;
    int top_count = (filled < l) ? filled : l;
    for (int i = 0; i < top_count; i++) {
        format_combo(best_stats[i].combo, best_stats[i].len, results[results_count].combination);
        // if we want subsets for display, build table quickly again or store them:
        // simpler approach: rebuild table once more
        // (some overhead but code is simpler; only matters if j is large)
        {
            SubsetTable* tmp_table = create_subset_table(HASH_SIZE);
            for (int d = 0; d < use_count; d++) {
                process_draw(&sorted_draws_data[d*6], d, k, tmp_table);
            }
            format_subsets(best_stats[i].combo, j, k, use_count, tmp_table, results[results_count].subsets);
            free_subset_table(tmp_table);
        }

        results[results_count].avg_rank = best_stats[i].avg_rank;
        results[results_count].min_value = best_stats[i].min_rank;
        results[results_count].is_chain_result = 0;
        // chain fields not used:
        results[results_count].draw_offset = 0;
        results[results_count].analysis_start_draw = 0;
        results[results_count].draws_until_common = 0;
        results_count++;
    }

    // plus n "selected" combos
    if (n > 0 && filled > top_count) {
        int remain = filled - top_count;
        int sel_count = (remain < n) ? remain : n;
        for (int i = 0; i < sel_count; i++) {
            int idx = top_count + i;
            format_combo(best_stats[idx].combo, best_stats[idx].len, results[results_count].combination);

            {
                SubsetTable* tmp_table = create_subset_table(HASH_SIZE);
                for (int d = 0; d < use_count; d++) {
                    process_draw(&sorted_draws_data[d*6], d, k, tmp_table);
                }
                format_subsets(best_stats[idx].combo, j, k, use_count, tmp_table, results[results_count].subsets);
                free_subset_table(tmp_table);
            }

            results[results_count].avg_rank = best_stats[idx].avg_rank;
            results[results_count].min_value = best_stats[idx].min_rank;
            results[results_count].is_chain_result = 0;
            results[results_count].draw_offset = 0;
            results[results_count].analysis_start_draw = 0;
            results[results_count].draws_until_common = 0;
            results_count++;
        }
    }

    free(best_stats);
    *out_len = results_count;
    if (results_count == 0) {
        free(results);
        return NULL;
    }
    return results;
}

// ----------------------------------------------------------------------
// Chain analysis (l = -1)
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
    // index i => sorted_draws_data[i*6..i*6+5]
    uint64* draw_patterns = (uint64*)malloc(draws_count * sizeof(uint64));
    if (!draw_patterns) {
        free(chain_results);
        *out_len = 0;
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        uint64 pat = 0ULL;
        for (int z = 0; z < 6; z++) {
            int val = sorted_draws_data[i*6 + z];
            pat |= (1ULL << (val - 1));
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

        // Build subset table from the newest use_count draws
        SubsetTable* table = create_subset_table(HASH_SIZE);
        for (int i = 0; i < use_count; i++) {
            process_draw(&sorted_draws_data[i*6], i, k, table);
        }

        // We want top-1 combo under j,k,m
        ComboStats best_stat;
        memset(&best_stat, 0, sizeof(best_stat));
        int found_any = 0;
        double best_val = -1e9; // for "max" tracking

        // Enumerate j-combos [1..j], same as standard
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

            if (!found_any) {
                found_any = 1;
                best_stat = stats;
                if (strcmp(m, "avg") == 0) {
                    best_val = stats.avg_rank;
                } else {
                    best_val = stats.min_rank;
                }
            } else {
                double val = (strcmp(m, "avg") == 0) ? stats.avg_rank : stats.min_rank;
                if (val > best_val) {
                    best_val = val;
                    best_stat = stats;
                }
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
            // no combos => break
            break;
        }

        // Fill chain_results item
        AnalysisResultItem* out_item = &chain_results[chain_index];
        format_combo(best_stat.combo, best_stat.len, out_item->combination);

        // Build a subsets string for display
        {
            SubsetTable* temp_t = create_subset_table(HASH_SIZE);
            for (int i = 0; i < use_count; i++) {
                process_draw(&sorted_draws_data[i*6], i, k, temp_t);
            }
            format_subsets(best_stat.combo, j, k, use_count, temp_t, out_item->subsets);
            free_subset_table(temp_t);
        }
        out_item->avg_rank = best_stat.avg_rank;
        out_item->min_value = best_stat.min_rank;
        out_item->is_chain_result = 1;
        out_item->draw_offset = chain_index + 1;  // "Analysis #"
        out_item->analysis_start_draw = draws_count - current_offset;  // "For Draw"

        // Now look for forward draws (offset-1, offset-2,...,0)
        // Checking if there's a common k-subset
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
            if (popcount(inter) >= k) {
                found_common = 1;
                break;
            }
        }
        // If we never found it, we do the "imagined future draw" correction:
        if (!found_common) {
            // pretend we found it after current_offset + 1
            i = current_offset + 1;
        } else if (i > current_offset) {
            i = current_offset;
        }

        // "Top-Ranked Duration" = i - 1  (from your definition)
        out_item->draws_until_common = (i > 0) ? (i - 1) : 0;

        // offset for next chain iteration
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

// ----------------------------------------------------------------------
// free_analysis_results(...)
// ----------------------------------------------------------------------
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

static inline int popcount(uint64 x) {
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
    // Simple 64 -> 32 hash
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
    double min_rank = total_draws;
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
            if (i > 0) out[pos++] = ',';
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
