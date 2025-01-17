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
// Internal data structures & precomputed tables
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

/**
 * Helper to run the "top-l" normal analysis (including l >= 1 or l=1).
 * This returns up to (l + n) results, same as the existing code.
 */
static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data, // copy of draws for subset table
    int use_count,
    int j,
    int k,
    const char* m,
    int l,
    int n,
    int max_number,
    int* out_len
);

/**
 * Helper to run the "chain" analysis if l == -1.
 * Each chain iteration calls a single top-1 analysis with the current offset,
 * then searches for the next offset by scanning forward draws.
 */
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
        return NULL; // safety check
    }
    init_tables();

    // Determine max_number from game_type
    // This is a small convenience, same as original code:
    int max_number;
    if (strstr(game_type, "6_49")) {
        max_number = 49;
    } else {
        max_number = 42;
    }

    // "use_count" is how many draws we use (the top portion).
    // For the standard approach, we do "draws_count - last_offset" draws.
    // For the chain approach, we'll pass that as well, but inside run_chain_analysis
    // each iteration deals with the offset. We'll unify by building a single array
    // of sorted draws for reference.
    int use_count = draws_count;
    if (use_count < 1) return NULL;

    // Make a local array "sorted_draws_data" holding each draw's 6 numbers, sorted.
    // We'll store them in ascending offset order: index 0 => offset=use_count-1, etc.
    // or we can store them in "time ascending" order. Because the chain logic
    // wants to scan forward draws from offset-1 down to 0. So let's store them
    // in ascending order: index 0 => oldest draw, index use_count-1 => newest draw.
    // Then an "offset" from the last draw means we pick index = use_count - 1 - offset.
    int* sorted_draws_data = (int*)malloc(use_count * 6 * sizeof(int));
    if (!sorted_draws_data) return NULL;

    // Copy and sort each draw row by ascending order
    // We'll store draw i in sorted_draws_data[i*6 + ...]
    // The DB code always orders draws by sort_order ascending (oldest first).
    // So index 0 is the oldest, index use_count-1 is the newest.
    for (int i = 0; i < use_count; i++) {
        // sort the 6 numbers
        int temp[6];
        for (int z = 0; z < 6; z++) {
            temp[z] = draws[i][z];
        }
        // bubble-sort them quickly or any method
        for (int a = 0; a < 5; a++) {
            for (int b = a + 1; b < 6; b++) {
                if (temp[a] > temp[b]) {
                    int tmpv = temp[a];
                    temp[a] = temp[b];
                    temp[b] = tmpv;
                }
            }
        }
        // store
        for (int z = 0; z < 6; z++) {
            sorted_draws_data[i*6 + z] = temp[z];
        }
    }

    // If not chain analysis (l != -1), do normal top-l approach
    if (l != -1) {
        // run standard approach
        AnalysisResultItem* ret = run_standard_analysis(
            sorted_draws_data,
            draws_count - last_offset, // we only "use" draws_count - last_offset
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
        j, k, m, max_number,
        out_len
    );
    free(sorted_draws_data);
    return chain_ret;
}

// ----------------------------------------------------------------------
// Standard (non-chain) top-l analysis, same as your original code’s logic
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
    // Edge cases
    if (use_count < 1) {
        *out_len = 0;
        return NULL;
    }

    // Build subset table from the newest 'use_count' draws
    // (Equivalent to the original code: "for i in range(use_count): process_draw(...)")
    SubsetTable* table = create_subset_table(HASH_SIZE);
    for (int i = 0; i < use_count; i++) {
        // draws are sorted_draws_data[i*6 .. i*6+5]
        process_draw(&sorted_draws_data[i*6], i, k, table);
    }

    // We'll find top-l combos by enumerating all j-combinations from [1..max_number].
    // This is the same logic from your original approach.
    // Implementation shortened for clarity, but it matches the code in your existing function.
    // We store them in a local "best_stats" array of size l. Then optionally fill up to n more.

    // We'll do a big "curr_combo" approach
    int capacity = l + n;
    AnalysisResultItem* results = (AnalysisResultItem*)calloc(capacity, sizeof(AnalysisResultItem));
    if (!results) {
        free_subset_table(table);
        return NULL;
    }

    // We'll store the best combos in a small array of ComboStats:
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
    // Initialize the first combination [1..j]
    for (int i = 0; i < j; i++) {
        curr_combo[i] = i + 1;
    }

    int filled = 0;
    while (1) {
        // Evaluate the current j-combo
        ComboStats stats;
        evaluate_combo(curr_combo, j, k, use_count, table, &stats);

        // Insert into best_stats if it beats the last
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
                    ComboStats tmp = best_stats[i];
                    best_stats[i] = best_stats[i-1];
                    best_stats[i-1] = tmp;
                } else break;
            }
        } else {
            // compare vs. best_stats[l-1]
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
                        ComboStats tmp = best_stats[i];
                        best_stats[i] = best_stats[i-1];
                        best_stats[i-1] = tmp;
                    } else break;
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

    // Fill in the results array
    int results_count = 0;
    // top-l combos
    int top_count = (filled < l) ? filled : l;
    for (int i = 0; i < top_count; i++) {
        format_combo(best_stats[i].combo, best_stats[i].len, results[results_count].combination);
        format_subsets(best_stats[i].combo, j, k, use_count, table, results[results_count].subsets);
        results[results_count].avg_rank = best_stats[i].avg_rank;
        results[results_count].min_value = best_stats[i].min_rank;
        results[results_count].is_chain_result = 0;
        // chain fields not used
        results[results_count].draw_offset = 0;
        results[results_count].analysis_start_draw = 0;
        results[results_count].draws_until_common = 0;
        results_count++;
    }
    // next n combos for "selected combos w/o overlapping subsets"
    // (mirroring original logic: just the next best combos up to n)
    if (n > 0 && filled > top_count) {
        int remain = filled - top_count;
        int sel_count = (remain < n) ? remain : n;
        for (int i = 0; i < sel_count; i++) {
            int idx = top_count + i;
            format_combo(best_stats[idx].combo, best_stats[idx].len, results[results_count].combination);
            format_subsets(best_stats[idx].combo, j, k, use_count, table, results[results_count].subsets);
            results[results_count].avg_rank = best_stats[idx].avg_rank;
            results[results_count].min_value = best_stats[idx].min_rank;
            results[results_count].is_chain_result = 0;
            results[results_count].draw_offset = 0;
            results[results_count].analysis_start_draw = 0;
            results[results_count].draws_until_common = 0;
            results_count++;
        }
    }

    *out_len = results_count;
    free(curr_combo);
    free(best_stats);
    free_subset_table(table);

    if (results_count == 0) {
        free(results);
        return NULL;
    }
    return results;
}

// ----------------------------------------------------------------------
// Chain analysis
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
    /**
     * We run repeated top-1 analyses at offset = current_offset,
     * then search forward draws from (current_offset-1) down to 0 for any
     * common k-subset. If found after i steps, draws_until_common = i-1, new offset = current_offset - i.
     * If not found at all, draws_until_common = (the total # steps) - 1, offset goes to 0 and we stop.
     */

    // We'll store chain results in a dynamic array. The chain length cannot exceed initial_offset+1.
    // For safety, we won't exceed draws_count either. We'll allocate enough.
    AnalysisResultItem* chain_results = (AnalysisResultItem*)calloc(initial_offset + 2, sizeof(AnalysisResultItem));
    if (!chain_results) {
        *out_len = 0;
        return NULL;
    }

    // Precompute bit patterns for each draw (for quick "common k-subset" detection).
    // sorted_draws_data[i*6..i*6+5] => bit pattern
    uint64* draw_patterns = (uint64*)malloc(draws_count * sizeof(uint64));
    if (!draw_patterns) {
        free(chain_results);
        *out_len = 0;
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        uint64 pattern = 0ULL;
        for (int z = 0; z < 6; z++) {
            int num = sorted_draws_data[i*6 + z];
            pattern |= (1ULL << (num - 1));
        }
        draw_patterns[i] = pattern;
    }

    int chain_index = 0;
    int current_offset = initial_offset;

    while (1) {
        if (current_offset < 0) break;  // done
        // We can't analyze if offset is bigger than draws_count - 1.
        if (current_offset > (draws_count - 1)) {
            // no more draws to use, so break
            break;
        }

        int use_count = draws_count - current_offset;
        if (use_count < 1) break;

        // 1) Build subset table from the newest 'use_count' draws
        SubsetTable* table = create_subset_table(HASH_SIZE);
        for (int i = 0; i < use_count; i++) {
            process_draw(&sorted_draws_data[i*6], i, k, table);
        }

        // 2) Run a top-1 analysis (l=1) with the same j, k, m
        //    We'll do a simpler version of the logic from run_standard_analysis (just top-1).
        int found_any = 0;
        ComboStats best_stat;
        memset(&best_stat, 0, sizeof(best_stat));
        double worst_val = -1e9; // or min_val
        int* curr_combo = (int*)malloc(j * sizeof(int));
        if (!curr_combo) {
            free_subset_table(table);
            break;
        }
        for (int z = 0; z < j; z++) {
            curr_combo[z] = z + 1;
        }

        while (1) {
            ComboStats stats;
            evaluate_combo(curr_combo, j, k, use_count, table, &stats);

            int better = 0;
            if (!found_any) {
                better = 1;
            } else {
                if (strcmp(m, "avg") == 0) {
                    if (stats.avg_rank > worst_val) {
                        better = 1;
                    }
                } else {
                    if (stats.min_rank > worst_val) {
                        better = 1;
                    }
                }
            }
            if (better) {
                memcpy(&best_stat, &stats, sizeof(ComboStats));
                if (strcmp(m, "avg") == 0) {
                    worst_val = stats.avg_rank;
                } else {
                    worst_val = stats.min_rank;
                }
                found_any = 1;
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

        // if we did not find any top-1 combo, break
        if (!found_any) {
            break;
        }

        // fill next chain result item
        AnalysisResultItem* item = &chain_results[chain_index];
        format_combo(best_stat.combo, best_stat.len, item->combination);
        // We'll also store subsets for debugging
        // Rebuild table or do the same approach? For clarity, let's rebuild again quickly:
        {
            SubsetTable* temp_table = create_subset_table(HASH_SIZE);
            for (int i = 0; i < use_count; i++) {
                process_draw(&sorted_draws_data[i*6], i, k, temp_table);
            }
            format_subsets(best_stat.combo, j, k, use_count, temp_table, item->subsets);
            free_subset_table(temp_table);
        }
        item->avg_rank = best_stat.avg_rank;
        item->min_value = best_stat.min_rank;
        item->is_chain_result = 1;

        // Analysis # => store in draw_offset
        item->draw_offset = chain_index + 1;
        // "For Draw" => total_draws - current_offset
        item->analysis_start_draw = draws_count - current_offset;

        // 3) Next, find the forward draws with offsets [current_offset - 1, ..., 0]
        //    searching for a common k-subset
        //    We'll check if popcount( combo_pattern & forward_draw_pattern ) >= k
        //    If so, stop.
        uint64 combo_pattern = 0ULL;
        for (int z = 0; z < j; z++) {
            combo_pattern |= (1ULL << (best_stat.combo[z] - 1));
        }

        int found_common = 0;
        int i;
        for (i = 1; i <= current_offset; i++) {
            // forward offset is (current_offset - i)
            int fidx = draws_count - 1 - (current_offset - i);
            // that is the index in sorted_draws_data (time ascending)
            if (fidx < 0) {
                break; // no more draws
            }
            if (fidx >= draws_count) {
                break;
            }
            // intersection?
            uint64 forward_pat = draw_patterns[fidx];
            uint64 inter = (combo_pattern & forward_pat);
            if (popcount(inter) >= k) {
                found_common = 1;
                break;
            }
        }

        // i is how many draws we advanced
        // If found_common=1, i is the count needed. If not found, i is how many we actually tested
        if (i > current_offset) {
            // means we tested all possible draws
            i = current_offset;
        }
        // "Top-Ranked Duration" => draws_until_common => i - 1
        item->draws_until_common = (i > 0) ? (i - 1) : 0;

        // new offset
        current_offset -= i;
        chain_index++;

        // If offset <= 0 or we found no more draws, break
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
// Free the results
// ----------------------------------------------------------------------
void free_analysis_results(AnalysisResultItem* results) {
    if (results) {
        free(results);
    }
}

// ----------------------------------------------------------------------
// Support functions
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
    // precompute bit counts
    for (int i = 0; i < 256; i++) {
        int cnt = 0;
        for (int j = 0; j < 8; j++) {
            if (i & (1 << j)) cnt++;
        }
        bit_count_table[i] = cnt;
    }
    initialized = 1;
}

static inline int popcount(uint64 x) {
    int c = 0;
    for (int i = 0; i < 8; i++) {
        c += bit_count_table[x & 0xFF];
        x >>= 8;
    }
    return c;
}

static SubsetTable* create_subset_table(int max_entries) {
    SubsetTable* t = (SubsetTable*)malloc(sizeof(SubsetTable));
    t->size = 0;
    t->capacity = max_entries;
    t->keys = (uint64*)calloc(max_entries, sizeof(uint64));
    t->values = (int*)malloc(max_entries * sizeof(int));
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
    // A variant of 64-bit -> 32-bit hashing
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
    // Enumerate all k-subsets from these 6 numbers and insert them with value=draw_idx
    // This is identical to your original approach.
    if (k > 6) return;
    int idx[20];
    for (int i = 0; i < k; i++) idx[i] = i; // choose first k
    while (1) {
        uint64 pattern = 0ULL;
        for (int i = 0; i < k; i++) {
            pattern |= (1ULL << (draw[idx[i]] - 1));
        }
        insert_subset(table, pattern, draw_idx);
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
        uint64 pattern = 0ULL;
        for (int i = 0; i < k; i++) {
            pattern |= (1ULL << (combo[idx[i]] - 1));
        }
        int last_seen = lookup_subset(table, pattern);
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
            idx[x] = idx[x - 1] + 1;
        }
    }

    stats->pattern = numbers_to_pattern(combo, j);
    stats->avg_rank = sum_ranks / (double)count;
    stats->min_rank = min_rank;
    memcpy(stats->combo, combo, j*sizeof(int));
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
        if (!first) {
            if (pos < MAX_SUBSETS_STR - 2) {
                out[pos++] = ',';
                out[pos++] = ' ';
            }
        }
        first = 0;
        if (pos >= MAX_SUBSETS_STR - 20) break;

        // subset
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

        uint64 pattern = 0ULL;
        for (int i = 0; i < k; i++) {
            pattern |= (1ULL << (combo[idx[i]] - 1));
        }
        int last_seen = lookup_subset(table, pattern);
        int rank = (last_seen >= 0) ? (total_draws - last_seen - 1) : total_draws;
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
