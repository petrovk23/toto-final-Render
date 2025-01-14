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
#define MAX_NUMBERS 50  // Max supported numbers (for 6/49 and 6/42)
#define LOOKUP_BITS 16  // For subset indexing optimization

typedef unsigned long long uint64;
typedef unsigned int uint32;

// Pre-computed lookup tables
static uint64 choose_table[MAX_NUMBERS][MAX_NUMBERS];
static int initialized = 0;

// Initialize combinatorial lookup table
static void init_lookup_tables() {
    if (initialized) return;
    memset(choose_table, 0, sizeof(choose_table));
    for (int n = 0; n < MAX_NUMBERS; n++) {
        choose_table[n][0] = 1;
        for (int k = 1; k <= n; k++) {
            choose_table[n][k] = choose_table[n-1][k-1] + choose_table[n-1][k];
        }
    }
    initialized = 1;
}

// Fast subset rank computation using bit manipulation
static uint64 subset_rank(const int* subset, int k, int max_number) {
    uint64 rank = 0;
    int prev = 0;
    for (int i = 0; i < k; i++) {
        int curr = subset[i] - 1;  // Convert to 0-based
        if (curr > prev) {
            for (int j = prev + 1; j < curr; j++) {
                rank += choose_table[max_number - j - 1][k - i - 1];
            }
        }
        prev = curr;
    }
    return rank;
}

// Fast bit manipulation for subset checking
static inline uint64 to_bitmask(const int* numbers, int count) {
    uint64 mask = 0;
    for (int i = 0; i < count; i++) {
        mask |= 1ULL << (numbers[i] - 1);
    }
    return mask;
}

// Structure to track subset occurrences efficiently
typedef struct {
    int* last_seen;        // When each subset was last seen
    int max_subsets;       // Total possible k-subsets
    int draws_count;       // Total draws being analyzed
    int k;                 // k-size for subsets
    int max_number;        // Maximum number (42 or 49)
} SubsetTracker;

static SubsetTracker* create_tracker(int max_number, int k, int draws_count) {
    SubsetTracker* t = (SubsetTracker*)malloc(sizeof(SubsetTracker));
    t->max_number = max_number;
    t->k = k;
    t->draws_count = draws_count;
    t->max_subsets = (int)choose_table[max_number][k];
    t->last_seen = (int*)malloc(t->max_subsets * sizeof(int));

    // Initialize all subsets as never seen
    for (int i = 0; i < t->max_subsets; i++) {
        t->last_seen[i] = draws_count;
    }
    return t;
}

static void free_tracker(SubsetTracker* t) {
    if (!t) return;
    free(t->last_seen);
    free(t);
}

// Track when each k-subset was last seen
static void update_tracker(SubsetTracker* t, const int* numbers, int draw_idx) {
    static int subset[20];
    int n = 6;  // numbers length (always 6 for TOTO)
    int k = t->k;

    // Generate all k-subsets of the numbers and update their last seen index
    int c[20];
    for (int i = 0; i < k; i++) c[i] = i;

    while (1) {
        for (int i = 0; i < k; i++) subset[i] = numbers[c[i]];

        uint64 rank = subset_rank(subset, k, t->max_number);
        if (rank < t->max_subsets &&
            (t->last_seen[rank] > draw_idx)) {
            t->last_seen[rank] = draw_idx;
        }

        // Generate next combination
        int i = k - 1;
        while (i >= 0 && c[i] == n - k + i) i--;
        if (i < 0) break;
        c[i]++;
        for (int j = i + 1; j < k; j++) c[j] = c[i] + j - i;
    }
}

// Structure to hold combo statistics
typedef struct {
    int combo[64];
    int combo_len;
    double avg_rank;
    double min_rank;
    int total_draws;
} ComboStats;

// Evaluate a single combination
static void evaluate_combo(
    const int* combo,
    int j,
    int k,
    SubsetTracker* t,
    ComboStats* stats
) {
    static int subset[20];
    long long sum_ranks = 0;
    double min_rank = t->draws_count + 1;
    int count = 0;

    // Generate all k-subsets of the combo
    int c[20];
    for (int i = 0; i < k; i++) c[i] = i;

    while (1) {
        for (int i = 0; i < k; i++) subset[i] = combo[c[i]];

        uint64 rank = subset_rank(subset, k, t->max_number);
        int last_seen = (rank < t->max_subsets) ? t->last_seen[rank] : t->draws_count;

        double rank_val = last_seen + 1;
        sum_ranks += rank_val;
        if (rank_val < min_rank) min_rank = rank_val;
        count++;

        // Generate next combination
        int i = k - 1;
        while (i >= 0 && c[i] == j - k + i) i--;
        if (i < 0) break;
        c[i]++;
        for (int j = i + 1; j < k; j++) c[j] = c[i] + j - i;
    }

    stats->avg_rank = (double)sum_ranks / count;
    stats->min_rank = min_rank;
    stats->total_draws = t->draws_count;
}

// Format subset information for display
static void format_subsets(
    const int* combo,
    int j,
    int k,
    SubsetTracker* t,
    char* out_buf
) {
    static int subset[20];
    char* ptr = out_buf;
    int remaining = MAX_SUBSETS_STR;

    ptr += snprintf(ptr, remaining, "[");
    remaining = MAX_SUBSETS_STR - (ptr - out_buf);

    int c[20];
    for (int i = 0; i < k; i++) c[i] = i;
    int first = 1;

    while (1 && remaining > 0) {
        for (int i = 0; i < k; i++) subset[i] = combo[c[i]];

        uint64 rank = subset_rank(subset, k, t->max_number);
        int last_seen = (rank < t->max_subsets) ? t->last_seen[rank] : t->draws_count;

        if (!first) {
            ptr += snprintf(ptr, remaining, ", ");
            remaining = MAX_SUBSETS_STR - (ptr - out_buf);
        }
        first = 0;

        ptr += snprintf(ptr, remaining, "((");
        for (int i = 0; i < k; i++) {
            ptr += snprintf(ptr, remaining, "%d%s", subset[i], i < k-1 ? "," : "");
        }
        ptr += snprintf(ptr, remaining, "), %d)", last_seen + 1);
        remaining = MAX_SUBSETS_STR - (ptr - out_buf);

        int i = k - 1;
        while (i >= 0 && c[i] == j - k + i) i--;
        if (i < 0) break;
        c[i]++;
        for (int j = i + 1; j < k; j++) c[j] = c[i] + j - i;
    }

    snprintf(ptr, remaining, "]");
}

static void combo_to_string(const int* combo, int len, char* out) {
    char* ptr = out;
    int remaining = MAX_COMBO_STR;

    for (int i = 0; i < len; i++) {
        int written = snprintf(ptr, remaining, "%d%s", combo[i], i < len-1 ? "," : "");
        ptr += written;
        remaining -= written;
        if (remaining <= 0) break;
    }
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
    if (!initialized) init_lookup_tables();

    if (j > MAX_ALLOWED_J) {
        fprintf(stderr, "Error: j=%d exceeds safety limit %d.\n", j, MAX_ALLOWED_J);
        *out_len = 0;
        return NULL;
    }

    int max_number = strstr(game_type, "6_49") ? 49 : 42;
    *out_len = 0;

    // Sort numbers within each draw
    for (int i = 0; i < draws_count; i++) {
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

    int use_count = draws_count - last_offset;
    if (use_count < 1) return NULL;

    // Create and populate subset tracker
    SubsetTracker* tracker = create_tracker(max_number, k, use_count);
    for (int i = 0; i < use_count; i++) {
        update_tracker(tracker, draws[i], i);
    }

    // Allocate space for results
    int capacity = (l == -1) ? 1000 : (l + n);
    AnalysisResultItem* results = (AnalysisResultItem*)calloc(capacity, sizeof(AnalysisResultItem));

    if (l == -1) {
        // Chain analysis
        int current_offset = last_offset;

        while (current_offset >= 0 && current_offset < draws_count) {
            int use_count = draws_count - current_offset;
            if (use_count < 1) break;

            // Find best combination
            ComboStats best_stats;
            memset(&best_stats, 0, sizeof(ComboStats));
            best_stats.min_rank = tracker->draws_count + 1;
            int best_combo[64];

            int* curr_combo = (int*)malloc(j * sizeof(int));
            for (int i = 0; i < j; i++) curr_combo[i] = i + 1;

            do {
                ComboStats stats;
                evaluate_combo(curr_combo, j, k, tracker, &stats);

                int better;
                if (strcmp(m, "avg") == 0) {
                    better = (stats.avg_rank < best_stats.avg_rank) ||
                            (best_stats.avg_rank == 0);
                } else {
                    better = (stats.min_rank < best_stats.min_rank) ||
                            (best_stats.min_rank == tracker->draws_count + 1);
                }

                if (better) {
                    memcpy(best_combo, curr_combo, j * sizeof(int));
                    best_stats = stats;
                }

                // Generate next combination
                int i = j - 1;
                while (i >= 0 && curr_combo[i] == max_number - j + i + 1) i--;
                if (i < 0) break;
                curr_combo[i]++;
                for (int x = i + 1; x < j; x++) curr_combo[x] = curr_combo[i] + x - i;

            } while (1);

            free(curr_combo);

            // Check for common subsets
            int draws_until_match = 0;
            int found = 0;

            for (int d = draws_count - use_count; d < draws_count && !found; d++) {
                draws_until_match++;

                // Check if any k-subset matches
                int c[20];
                for (int i = 0; i < k; i++) c[i] = i;

                while (!found && c[0] <= j-k) {
                    int subset[20];
                    for (int i = 0; i < k; i++) subset[i] = best_combo[c[i]];

                    uint64 subset_mask = to_bitmask(subset, k);
                    uint64 draw_mask = to_bitmask(draws[d], 6);

                    if (__builtin_popcountll(subset_mask & draw_mask) == k) {
                        found = 1;
                        break;
                    }

                    // Next k-combination of j numbers
                    int i = k - 1;
                    while (i >= 0 && c[i] == j - k + i) i--;
                    if (i < 0) break;
                    c[i]++;
                    for (int x = i + 1; x < k; x++) c[x] = c[i] + x - i;
                }
            }

            if (!found) draws_until_match = use_count;

            // Store result
            AnalysisResultItem* outR = &results[*out_len];
            outR->is_chain_result = 1;
            outR->draw_offset = current_offset;
            outR->analysis_start_draw = draws_count - current_offset;
            outR->draws_until_common = draws_until_match - 1;

            combo_to_string(best_combo, j, outR->combination);
            outR->avg_rank = best_stats.avg_rank;
            outR->min_value = best_stats.min_rank;
            format_subsets(best_combo, j, k, tracker, outR->subsets);

            (*out_len)++;

            if (!found) break;
            current_offset = current_offset - draws_until_match;
        }
    } else {
        // Normal analysis
        ComboStats* best_combos = (ComboStats*)malloc(l * sizeof(ComboStats));
        int filled = 0;

        int* curr_combo = (int*)malloc(j * sizeof(int));
        for (int i = 0; i < j; i++) curr_combo[i] = i + 1;

        do {
            ComboStats stats;
            evaluate_combo(curr_combo, j, k, tracker, &stats);

            if (filled < l) {
                memcpy(best_combos[filled].combo, curr_combo, j * sizeof(int));
                best_combos[filled].combo_len = j;
                best_combos[filled].avg_rank = stats.avg_rank;
                best_combos[filled].min_rank = stats.min_rank;
                best_combos[filled].total_draws = stats.total_draws;
                filled++;

                // Sort by descending order
                for (int i = filled - 1; i > 0; i--) {
                    int should_swap = 0;
                    if (strcmp(m, "avg") == 0) {
                        should_swap = best_combos[i].avg_rank > best_combos[i-1].avg_rank;
                    } else {
                        should_swap = best_combos[i].min_rank > best_combos[i-1].min_rank;
                    }

                    if (should_swap) {
                        ComboStats tmp = best_combos[i];
                        best_combos[i] = best_combos[i-1];
                        best_combos[i-1] = tmp;
                    } else break;
                }
            } else {
                int should_replace = 0;
                if (strcmp(m, "avg") == 0) {
                    should_replace = stats.avg_rank > best_combos[l-1].avg_rank;
                } else {
                    should_replace = stats.min_rank > best_combos[l-1].min_rank;
                }

                if (should_replace) {
                    best_combos[l-1].avg_rank = stats.avg_rank;
                    best_combos[l-1].min_rank = stats.min_rank;
                    best_combos[l-1].total_draws = stats.total_draws;
                    memcpy(best_combos[l-1].combo, curr_combo, j * sizeof(int));
                    best_combos[l-1].combo_len = j;

                    // Bubble up
                    for (int i = l-1; i > 0; i--) {
                        int should_swap = 0;
                        if (strcmp(m, "avg") == 0) {
                            should_swap = best_combos[i].avg_rank > best_combos[i-1].avg_rank;
                        } else {
                            should_swap = best_combos[i].min_rank > best_combos[i-1].min_rank;
                        }

                        if (should_swap) {
                            ComboStats tmp = best_combos[i];
                            best_combos[i] = best_combos[i-1];
                            best_combos[i-1] = tmp;
                        } else break;
                    }
                }
            }

            // Generate next combination
            int i = j - 1;
            while (i >= 0 && curr_combo[i] == max_number - j + i + 1) i--;
            if (i < 0) break;
            curr_combo[i]++;
            for (int x = i + 1; x < j; x++) curr_combo[x] = curr_combo[i] + x - i;

        } while (1);

        free(curr_combo);

        // Store results
        for (int i = 0; i < filled; i++) {
            AnalysisResultItem* outR = &results[*out_len];
            outR->is_chain_result = 0;

            combo_to_string(best_combos[i].combo, best_combos[i].combo_len, outR->combination);
            outR->avg_rank = best_combos[i].avg_rank;
            outR->min_value = best_combos[i].min_rank;
            format_subsets(best_combos[i].combo, j, k, tracker, outR->subsets);

            (*out_len)++;
        }

        // Handle non-overlapping combos if requested
        if (n > 0) {
            for (int i = 0; i < filled && *out_len < capacity; i++) {
                AnalysisResultItem* outR = &results[*out_len];
                outR->is_chain_result = 0;

                combo_to_string(best_combos[i].combo, best_combos[i].combo_len, outR->combination);
                outR->avg_rank = best_combos[i].avg_rank;
                outR->min_value = best_combos[i].min_rank;
                format_subsets(best_combos[i].combo, j, k, tracker, outR->subsets);

                (*out_len)++;
            }
        }

        free(best_combos);
    }

    free_tracker(tracker);

    if (*out_len > MAX_ALLOWED_OUT_LEN) {
        fprintf(stderr, "Error: out_len=%d exceeds safety limit %d.\n", *out_len, MAX_ALLOWED_OUT_LEN);
        free(results);
        results = NULL;
        *out_len = 0;
    }

    return results;
}

void free_analysis_results(AnalysisResultItem* results) {
    if (results) {
        free(results);
    }
}
