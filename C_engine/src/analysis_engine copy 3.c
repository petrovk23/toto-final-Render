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
    int* last_seen;        // When each subset was last seen (0 = most recent draw)
    int max_subsets;       // Total possible k-subsets
    int draws_count;       // Total draws being analyzed
    int k;                 // k-size for subsets
    int max_number;        // Maximum number (42 or 49)
} SubsetTracker;

static SubsetTracker* create_tracker(int max_number, int k, int use_count) {
    SubsetTracker* t = (SubsetTracker*)malloc(sizeof(SubsetTracker));
    t->max_number = max_number;
    t->k = k;
    t->draws_count = use_count;  // We only consider 'use_count' draws
    t->max_subsets = (int)choose_table[max_number][k];
    t->last_seen = (int*)malloc(t->max_subsets * sizeof(int));

    // Initialize all subsets as never seen => rank = use_count
    for (int i = 0; i < t->max_subsets; i++) {
        t->last_seen[i] = use_count;
    }
    return t;
}

static void free_tracker(SubsetTracker* t) {
    if (!t) return;
    free(t->last_seen);
    free(t);
}

// Track when each k-subset was last seen, in "draw_age" (0 = newest draw)
static void update_tracker(SubsetTracker* t, const int* numbers, int draw_age) {
    static int subset[20];
    int n = 6;  // TOTO draws have 6 numbers
    int k = t->k;

    int c[20];
    for (int i = 0; i < k; i++) c[i] = i;

    while (1) {
        for (int i = 0; i < k; i++) {
            subset[i] = numbers[c[i]];
        }
        uint64 rank = subset_rank(subset, k, t->max_number);
        if (rank < (uint64)t->max_subsets) {
            // If this draw_age is more recent (smaller) than we had,
            // update.  (0 = newest; larger = older.)
            if (draw_age < t->last_seen[rank]) {
                t->last_seen[rank] = draw_age;
            }
        }

        // Next combination
        int i = k - 1;
        while (i >= 0 && c[i] == n - k + i) i--;
        if (i < 0) break;
        c[i]++;
        for (int j = i + 1; j < k; j++) {
            c[j] = c[i] + j - i;
        }
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
    double min_rank = t->draws_count + 0.0;  // something large
    int count = 0;

    // Generate all k-subsets of the combo
    int c[20];
    for (int i = 0; i < k; i++) c[i] = i;

    while (1) {
        for (int i = 0; i < k; i++) {
            subset[i] = combo[c[i]];
        }

        uint64 rank = subset_rank(subset, k, t->max_number);
        int last_seen = (rank < (uint64)t->max_subsets)
                        ? t->last_seen[rank]
                        : t->draws_count;

        // rank_val = 0 => last draw, 1 => second-last, etc.
        double rank_val = (double)last_seen;
        sum_ranks += rank_val;
        if (rank_val < min_rank) {
            min_rank = rank_val;
        }
        count++;

        // Next combination
        int i = k - 1;
        while (i >= 0 && c[i] == j - k + i) i--;
        if (i < 0) break;
        c[i]++;
        for (int x = i + 1; x < k; x++) {
            c[x] = c[i] + x - i;
        }
    }

    stats->avg_rank = (count == 0) ? 0.0 : ((double)sum_ranks / (double)count);
    stats->min_rank = (count == 0) ? 0.0 : min_rank;
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
        int last_seen = (rank < (uint64)t->max_subsets)
                        ? t->last_seen[rank]
                        : t->draws_count;

        if (!first) {
            ptr += snprintf(ptr, remaining, ", ");
            remaining = MAX_SUBSETS_STR - (ptr - out_buf);
        }
        first = 0;

        ptr += snprintf(ptr, remaining, "((");
        remaining = MAX_SUBSETS_STR - (ptr - out_buf);

        for (int i = 0; i < k && remaining > 0; i++) {
            ptr += snprintf(ptr, remaining, "%d", subset[i]);
            remaining = MAX_SUBSETS_STR - (ptr - out_buf);
            if (i < k - 1 && remaining > 0) {
                ptr += snprintf(ptr, remaining, ",");
                remaining = MAX_SUBSETS_STR - (ptr - out_buf);
            }
        }
        if (remaining > 0) {
            ptr += snprintf(ptr, remaining, "), %d)", last_seen);
            remaining = MAX_SUBSETS_STR - (ptr - out_buf);
        }

        // Next k-combo
        int i = k - 1;
        while (i >= 0 && c[i] == j - k + i) i--;
        if (i < 0) break;
        c[i]++;
        for (int x = i + 1; x < k; x++) {
            c[x] = c[i] + x - i;
        }
    }

    if (remaining > 0) {
        snprintf(ptr, remaining, "]");
    }
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

    // Distinguish if 6_49 or 6_42
    int max_number = strstr(game_type, "6_49") ? 49 : 42;
    *out_len = 0;

    // Sort numbers ascending in each draw
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

    // Create and populate subset tracker with 'use_count' draws,
    // from newest to oldest so that draw_age=0 => the last row.
    SubsetTracker* tracker = create_tracker(max_number, k, use_count);

    // We read from draws_count-1 down to draws_count-use_count
    // and assign age=0 for the last draw, 1 for the one before, etc.
    for (int r = 0; r < use_count; r++) {
        int real_idx = draws_count - 1 - r;
        update_tracker(tracker, draws[real_idx], r);
    }

    // Allocate space for results
    int capacity = (l == -1) ? 1000 : (l + n);
    AnalysisResultItem* results = (AnalysisResultItem*)calloc(capacity, sizeof(AnalysisResultItem));

    if (l == -1) {
        // -------------------------------
        // Chain analysis
        // -------------------------------
        int current_offset = last_offset;

        while (current_offset >= 0 && current_offset < draws_count) {
            int remaining_draws = draws_count - current_offset;
            if (remaining_draws < 1) break;

            // Find best combination
            ComboStats best_stats;
            memset(&best_stats, 0, sizeof(ComboStats));
            // For descending order, we start with extremely small values
            best_stats.avg_rank = -999999.0;
            best_stats.min_rank = -999999.0;

            int best_combo[64];

            // Brute force over j-combinations from [1..max_number]
            int* curr_combo = (int*)malloc(j * sizeof(int));
            for (int x = 0; x < j; x++) {
                curr_combo[x] = x + 1;
            }

            while (1) {
                ComboStats stats;
                evaluate_combo(curr_combo, j, k, tracker, &stats);

                // For descending order:
                // 'avg' => bigger avg_rank is better
                // 'min' => bigger min_rank is better
                int better;
                if (strcmp(m, "avg") == 0) {
                    better = (stats.avg_rank > best_stats.avg_rank);
                } else {
                    better = (stats.min_rank > best_stats.min_rank);
                }

                if (better) {
                    memcpy(best_combo, curr_combo, j * sizeof(int));
                    best_stats = stats;
                }

                // Next j-combination
                int i = j - 1;
                while (i >= 0 && curr_combo[i] == max_number - j + i + 1) i--;
                if (i < 0) break;
                curr_combo[i]++;
                for (int v = i + 1; v < j; v++) {
                    curr_combo[v] = curr_combo[i] + v - i;
                }
            }
            free(curr_combo);

            // Check how many draws until we see a common k-subset
            int draws_until_match = 0;
            int found = 0;

            for (int r = 0; r < remaining_draws; r++) {
                draws_until_match++;
                int real_idx = draws_count - 1 - current_offset - r;
                if (real_idx < 0) break;

                // Compare k-subsets
                int c[20];
                for (int i = 0; i < k; i++) c[i] = i;
                while (!found) {
                    int subset[20];
                    for (int i = 0; i < k; i++) {
                        subset[i] = best_combo[c[i]];
                    }
                    uint64 subset_mask = to_bitmask(subset, k);
                    uint64 draw_mask = to_bitmask(draws[real_idx], 6);

                    if (__builtin_popcountll(subset_mask & draw_mask) == k) {
                        found = 1;
                        break;
                    }
                    int i = k - 1;
                    while (i >= 0 && c[i] == j - k + i) i--;
                    if (i < 0) break;
                    c[i]++;
                    for (int x = i + 1; x < k; x++) {
                        c[x] = c[i] + x - i;
                    }
                }
                if (found) break;
            }
            if (!found) {
                draws_until_match = remaining_draws;
            }

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
            if (*out_len >= capacity) break;

            if (!found) {
                // No match => stop chain
                break;
            }
            current_offset = current_offset - draws_until_match;
        }
    } else {
        // -------------------------------
        // Normal analysis
        // -------------------------------
        ComboStats* best_combos = (ComboStats*)malloc(l * sizeof(ComboStats));
        int filled = 0;

        int* curr_combo = (int*)malloc(j * sizeof(int));
        for (int x = 0; x < j; x++) {
            curr_combo[x] = x + 1;
        }

        while (1) {
            ComboStats stats;
            evaluate_combo(curr_combo, j, k, tracker, &stats);

            if (filled < l) {
                // Place in best_combos
                memcpy(best_combos[filled].combo, curr_combo, j * sizeof(int));
                best_combos[filled].combo_len = j;
                best_combos[filled].avg_rank = stats.avg_rank;
                best_combos[filled].min_rank = stats.min_rank;
                best_combos[filled].total_draws = stats.total_draws;
                filled++;

                // Bubble up (descending order)
                for (int i = filled - 1; i > 0; i--) {
                    int should_swap;
                    if (strcmp(m, "avg") == 0) {
                        // bigger avg_rank => better
                        should_swap = (best_combos[i].avg_rank > best_combos[i - 1].avg_rank);
                    } else {
                        // bigger min_rank => better
                        should_swap = (best_combos[i].min_rank > best_combos[i - 1].min_rank);
                    }
                    if (should_swap) {
                        ComboStats tmp = best_combos[i];
                        best_combos[i] = best_combos[i - 1];
                        best_combos[i - 1] = tmp;
                    } else {
                        break;
                    }
                }
            } else {
                // Compare with worst stored (worst = end of array, because it's sorted descending)
                int should_replace;
                if (strcmp(m, "avg") == 0) {
                    should_replace = (stats.avg_rank > best_combos[l - 1].avg_rank);
                } else {
                    should_replace = (stats.min_rank > best_combos[l - 1].min_rank);
                }

                if (should_replace) {
                    best_combos[l - 1].avg_rank = stats.avg_rank;
                    best_combos[l - 1].min_rank = stats.min_rank;
                    best_combos[l - 1].total_draws = stats.total_draws;
                    memcpy(best_combos[l - 1].combo, curr_combo, j * sizeof(int));
                    best_combos[l - 1].combo_len = j;

                    // Bubble up
                    for (int i = l - 1; i > 0; i--) {
                        int should_swap;
                        if (strcmp(m, "avg") == 0) {
                            should_swap = (best_combos[i].avg_rank > best_combos[i - 1].avg_rank);
                        } else {
                            should_swap = (best_combos[i].min_rank > best_combos[i - 1].min_rank);
                        }
                        if (should_swap) {
                            ComboStats tmp = best_combos[i];
                            best_combos[i] = best_combos[i - 1];
                            best_combos[i - 1] = tmp;
                        } else {
                            break;
                        }
                    }
                }
            }

            // Next j-combination
            int i = j - 1;
            while (i >= 0 && curr_combo[i] == max_number - j + i + 1) i--;
            if (i < 0) break;
            curr_combo[i]++;
            for (int v = i + 1; v < j; v++) {
                curr_combo[v] = curr_combo[i] + v - i;
            }
        }
        free(curr_combo);

        // Store the top-l combos in results
        for (int i = 0; i < filled && i < l && *out_len < capacity; i++) {
            AnalysisResultItem* outR = &results[*out_len];
            outR->is_chain_result = 0;

            combo_to_string(best_combos[i].combo, best_combos[i].combo_len, outR->combination);
            outR->avg_rank = best_combos[i].avg_rank;
            outR->min_value = best_combos[i].min_rank;
            format_subsets(best_combos[i].combo, j, k, tracker, outR->subsets);

            (*out_len)++;
        }

        // If user wants up to n more combos with no overlap (or whatever custom logic),
        // we place up to n combos after that. For simplicity, we re-list combos here.
        if (n > 0) {
            for (int i = 0; i < filled && *out_len < capacity && i < n + l; i++) {
                AnalysisResultItem* outR = &results[*out_len];
                outR->is_chain_result = 0;

                combo_to_string(best_combos[i].combo, best_combos[i].combo_len, outR->combination);
                outR->avg_rank = best_combos[i].avg_rank;
                outR->min_value = best_combos[i].min_rank;
                format_subsets(best_combos[i].combo, j, k, tracker, outR->subsets);

                (*out_len)++;
                if (*out_len >= capacity) break;
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
