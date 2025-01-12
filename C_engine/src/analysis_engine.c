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
#define CACHE_LINE 64

typedef unsigned long long uint64;
typedef unsigned int uint32;

// Optimized subset hash table
typedef struct {
    uint64* keys;     // Subset bit patterns
    int* values;      // Last occurrence
    int size;         // Current size
    int capacity;     // Max capacity
} SubsetTable;

// Pre-computed combinatorial numbers
static uint64 nCk_table[MAX_NUMBERS][MAX_NUMBERS];
// Pre-computed bit counts
static int bit_count_table[256];
static int initialized = 0;

// SIMD-friendly combo stats
typedef struct __attribute__((aligned(16))) {
    uint64 pattern;   // Bit pattern of combo
    double avg_rank;
    double min_rank;
    int combo[MAX_NUMBERS];
    int len;
} ComboStats;

// Initialize lookup tables
static void init_tables() {
    if (initialized) return;

    // Compute nCk table
    memset(nCk_table, 0, sizeof(nCk_table));
    for (int n = 0; n < MAX_NUMBERS; n++) {
        nCk_table[n][0] = 1;
        for (int k = 1; k <= n; k++) {
            nCk_table[n][k] = nCk_table[n-1][k-1] + nCk_table[n-1][k];
        }
    }

    // Pre-compute bit count table
    for (int i = 0; i < 256; i++) {
        int count = 0;
        for (int j = 0; j < 8; j++) {
            if (i & (1 << j)) count++;
        }
        bit_count_table[i] = count;
    }

    initialized = 1;
}

// Fast popcount using lookup table
static inline int popcount(uint64 x) {
    int count = 0;
    for (int i = 0; i < 8; i++) {
        count += bit_count_table[x & 0xFF];
        x >>= 8;
    }
    return count;
}

// Create subset hash table
static SubsetTable* create_subset_table(int max_entries) {
    SubsetTable* table = (SubsetTable*)malloc(sizeof(SubsetTable));
    table->size = 0;
    table->capacity = max_entries;

    table->keys = (uint64*)calloc(max_entries, sizeof(uint64));
    table->values = (int*)malloc(max_entries * sizeof(int));
    memset(table->values, -1, max_entries * sizeof(int));

    return table;
}

static void free_subset_table(SubsetTable* table) {
    if (!table) return;
    free(table->keys);
    free(table->values);
    free(table);
}

// Fast hash function for subsets
static inline uint32 hash_subset(uint64 pattern) {
    pattern ^= pattern >> 33;
    pattern *= 0xff51afd7ed558ccdull;
    pattern ^= pattern >> 33;
    pattern *= 0xc4ceb9fe1a85ec53ull;
    pattern ^= pattern >> 33;
    return (uint32)(pattern & (HASH_SIZE - 1));
}

// Insert subset into hash table
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

// Look up subset in hash table
static inline int lookup_subset(const SubsetTable* table, uint64 pattern) {
    uint32 idx = hash_subset(pattern);
    while (1) {
        if (table->values[idx] == -1) return -1;
        if (table->keys[idx] == pattern) return table->values[idx];
        idx = (idx + 1) & (HASH_SIZE - 1);
    }
}

// Convert numbers to bit pattern
static inline uint64 numbers_to_pattern(const int* numbers, int count) {
    uint64 pattern = 0;
    for (int i = 0; i < count; i++) {
        pattern |= 1ULL << (numbers[i] - 1);
    }
    return pattern;
}

// Process draw and update subset table
static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table) {
    int indices[20] = {0};
    for (int i = 0; i < k; i++) indices[i] = i;

    do {
        uint64 pattern = 0;
        for (int i = 0; i < k; i++) {
            pattern |= 1ULL << (draw[indices[i]] - 1);
        }

        insert_subset(table, pattern, draw_idx);

        // Next k-combination
        int i = k - 1;
        while (i >= 0 && indices[i] == 6 - k + i) i--;
        if (i < 0) break;

        indices[i]++;
        for (int j = i + 1; j < k; j++) {
            indices[j] = indices[i] + (j - i);
        }
    } while (1);
}

// Evaluate a combination
static void evaluate_combo(const int* combo, int j, int k, int total_draws,
                         const SubsetTable* table, ComboStats* stats) {
    double sum_ranks = 0;
    double min_rank = total_draws;
    int count = 0;

    int indices[20] = {0};
    for (int i = 0; i < k; i++) indices[i] = i;

    do {
        uint64 pattern = 0;
        for (int i = 0; i < k; i++) {
            pattern |= 1ULL << (combo[indices[i]] - 1);
        }

        int last_seen = lookup_subset(table, pattern);
        double rank = (last_seen >= 0) ? total_draws - last_seen - 1 : total_draws;

        sum_ranks += rank;
        if (rank < min_rank) min_rank = rank;
        count++;

        // Next k-combination
        int i = k - 1;
        while (i >= 0 && indices[i] == j - k + i) i--;
        if (i < 0) break;

        indices[i]++;
        for (int x = i + 1; x < k; x++) {
            indices[x] = indices[i] + (x - i);
        }
    } while (1);

    stats->pattern = numbers_to_pattern(combo, j);
    stats->avg_rank = sum_ranks / count;
    stats->min_rank = min_rank;
    memcpy(stats->combo, combo, j * sizeof(int));
    stats->len = j;
}

static void format_combo(const int* combo, int len, char* out) {
    int pos = 0;
    for (int i = 0; i < len; i++) {
        if (i > 0) out[pos++] = ',';
        pos += sprintf(out + pos, "%d", combo[i]);
    }
    out[pos] = '\0';
}

static void format_subsets(const int* combo, int j, int k, int total_draws,
                         const SubsetTable* table, char* out) {
    int pos = 0;
    out[pos++] = '[';

    int indices[20] = {0};
    for (int i = 0; i < k; i++) indices[i] = i;

    int first = 1;
    do {
        if (!first) {
            out[pos++] = ',';
            out[pos++] = ' ';
        }
        first = 0;

        out[pos++] = '(';
        out[pos++] = '(';

        for (int i = 0; i < k; i++) {
            if (i > 0) out[pos++] = ',';
            pos += sprintf(out + pos, "%d", combo[indices[i]]);
        }

        out[pos++] = ')';
        out[pos++] = ',';
        out[pos++] = ' ';

        uint64 pattern = 0;
        for (int i = 0; i < k; i++) {
            pattern |= 1ULL << (combo[indices[i]] - 1);
        }

        int last_seen = lookup_subset(table, pattern);
        int rank = (last_seen >= 0) ? total_draws - last_seen - 1 : total_draws;

        pos += sprintf(out + pos, "%d)", rank);

        // Next k-combination
        int i = k - 1;
        while (i >= 0 && indices[i] == j - k + i) i--;
        if (i < 0) break;

        indices[i]++;
        for (int x = i + 1; x < k; x++) {
            indices[x] = indices[i] + (x - i);
        }

        if (pos > MAX_SUBSETS_STR - 50) break;
    } while (1);

    out[pos++] = ']';
    out[pos] = '\0';
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

    int max_number = strstr(game_type, "6_49") ? 49 : 42;
    init_tables();

    int use_count = draws_count - last_offset;
    if (use_count < 1) return NULL;

    // Sort numbers within draws
    for (int i = 0; i < use_count; i++) {
        for (int a = 0; a < 5; a++) {
            for (int b = a + 1; b < 6; b++) {
                if (draws[i][a] > draws[i][b]) {
                    int tmp = draws[i][a];
                    draws[i][a] = draws[i][b];
                    draws[i][b] = tmp;
                }
            }
        }
    }

    // Create and populate subset table
    SubsetTable* table = create_subset_table(HASH_SIZE);
    for (int i = 0; i < use_count; i++) {
        process_draw(draws[i], i, k, table);
    }

    // Allocate space for results and working memory
    int capacity = l + n;
    AnalysisResultItem* results = calloc(capacity, sizeof(AnalysisResultItem));
    if (!results) {
        free_subset_table(table);
        return NULL;
    }

    int* curr_combo = malloc(j * sizeof(int));
    ComboStats* best_stats = malloc(l * sizeof(ComboStats));
    if (!curr_combo || !best_stats) {
        free(curr_combo);
        free(best_stats);
        free(results);
        free_subset_table(table);
        return NULL;
    }

    // Initialize first combination
    for (int i = 0; i < j; i++) curr_combo[i] = i + 1;

    int filled = 0;
    do {
        ComboStats stats;
        evaluate_combo(curr_combo, j, k, use_count, table, &stats);

        if (filled < l) {
            memcpy(&best_stats[filled], &stats, sizeof(ComboStats));
            filled++;

            // Keep sorted in descending order
            for (int i = filled - 1; i > 0; i--) {
                int swap;
                if (strcmp(m, "avg") == 0) {
                    swap = best_stats[i].avg_rank > best_stats[i-1].avg_rank;
                } else {
                    swap = best_stats[i].min_rank > best_stats[i-1].min_rank;
                }

                if (swap) {
                    ComboStats tmp = best_stats[i];
                    best_stats[i] = best_stats[i-1];
                    best_stats[i-1] = tmp;
                } else break;
            }
        } else {
            int replace;
            if (strcmp(m, "avg") == 0) {
                replace = stats.avg_rank > best_stats[l-1].avg_rank;
            } else {
                replace = stats.min_rank > best_stats[l-1].min_rank;
            }

            if (replace) {
                best_stats[l-1] = stats;

                // Bubble up
                for (int i = l - 1; i > 0; i--) {
                    int swap;
                    if (strcmp(m, "avg") == 0) {
                        swap = best_stats[i].avg_rank > best_stats[i-1].avg_rank;
                    } else {
                        swap = best_stats[i].min_rank > best_stats[i-1].min_rank;
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
        for (int i = pos + 1; i < j; i++) {
            curr_combo[i] = curr_combo[pos] + (i - pos);
        }
    } while (1);

    // Fill results array
    int results_count = 0;
    for (int i = 0; i < filled && results_count < l; i++) {
        AnalysisResultItem* item = &results[results_count++];

        format_combo(best_stats[i].combo, best_stats[i].len, item->combination);
        format_subsets(best_stats[i].combo, j, k, use_count, table, item->subsets);

        item->avg_rank = best_stats[i].avg_rank;
        item->min_value = best_stats[i].min_rank;
        item->is_chain_result = 0;
    }

    // Handle "selected" results if requested
    if (n > 0) {
        for (int i = l; i < filled && results_count < l + n; i++) {
            AnalysisResultItem* item = &results[results_count++];

           format_combo(best_stats[i].combo, best_stats[i].len, item->combination);
           format_subsets(best_stats[i].combo, j, k, use_count, table, item->subsets);

           item->avg_rank = best_stats[i].avg_rank;
           item->min_value = best_stats[i].min_rank;
           item->is_chain_result = 0;
       }
   }

   // Clean up
   free(curr_combo);
   free(best_stats);
   free_subset_table(table);

   if (results_count == 0) {
       free(results);
       return NULL;
   }

   *out_len = results_count;
   return results;
}

void free_analysis_results(AnalysisResultItem* results) {
   free(results);
}
