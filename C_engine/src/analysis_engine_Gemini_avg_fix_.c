#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h> // Required for DBL_MIN

#include "analysis_engine.h"

#define MAX_COMBO_STR 255
#define MAX_SUBSETS_STR 65535
#define MAX_ALLOWED_J 20 // Reduced slightly for practical C(n,k) limits if needed, but keep as is for now.
#define MAX_NUMBERS 50     // Max number in any game (e.g., 49 for 6/49)
#define HASH_SIZE (1 << 26) // 67M entries - good balance
#define MAX_K_FOR_SUBSET_EVAL 10 // k > 10 might be too slow for subset iteration

typedef unsigned long long uint64;
typedef unsigned int uint32;

// ----------------------------------------------------------------------
// Internal data and lookups
// ----------------------------------------------------------------------
static uint64 nCk_table[MAX_NUMBERS + 1][MAX_K_FOR_SUBSET_EVAL + 1]; // Adjusted size
static int initialized = 0;

// Hash Table for storing last seen index of k-subsets
typedef struct {
    uint64* keys;     // Subset bit patterns
    int* values;      // Last occurrence index (0 to use_count-1)
    // No size/capacity needed for this simple linear probing implementation
} SubsetTable;

// Holds stats for a complete j-combination during backtracking/search
typedef struct {
    uint64 pattern;   // bit pattern of combo (1ULL << (num-1))
    double avg_rank;
    double min_rank;
    int combo[MAX_ALLOWED_J]; // Store the combination itself
    int len; // Should always be 'j' when fully evaluated
} ComboStats;

// ----------------------------------------------------------------------
// Forward declarations
// ----------------------------------------------------------------------
static void init_tables();
static inline int popcount64(uint64 x);
static SubsetTable* create_subset_table();
static void free_subset_table(SubsetTable* table);
static inline uint32 hash_subset(uint64 pattern);
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value);
static inline int lookup_subset(const SubsetTable* table, uint64 pattern);
static inline uint64 numbers_to_pattern(const int* numbers, int count);
static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table);
static void format_combo(const int* combo, int len, char* out);
static void format_subsets(const int* combo, int j, int k, int total_draws,
                           const SubsetTable* table, char* out);

// Main analysis functions
static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data,
    int use_count, // number of draws to use (draws_count - last_offset)
    int j, int k, const char* m, int l, int n,
    int max_number,
    int* out_len
);

static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data,
    int draws_count, // total number of draws available
    int initial_offset, // starting offset for the chain
    int j, int k, const char* m,
    int max_number,
    int* out_len
);

// Backtracking function
static void backtrack(
    // Current state
    int* S,             // The current partial combination being built
    int size,           // Current size of S (from 0 to j)
    uint64 current_S,   // Bit pattern of S
    double current_min_rank, // Minimum rank found amongst all subsets evaluated *so far* for this branch
    double sum_ranks_so_far, // Sum of ranks of all subsets evaluated *so far* for this branch
    int num_subsets_so_far, // Count of subsets evaluated *so far* for this branch
    int start_num,      // Next number to try adding to S

    // Context / Parameters
    const SubsetTable* table, // Hash table for subset lookups
    int total_draws,    // == use_count for standard, current use_count in chain
    int max_number,     // Max number allowed in combos (e.g., 42 or 49)
    int j, int k,
    uint64 Cjk,         // Total number of k-subsets in a j-combo = C(j, k)

    // Top-L management (thread-local)
    ComboStats* thread_best, // Array to hold top L combos for this thread
    int* thread_filled, // Number of slots filled in thread_best (0 to l)
    int l,              // Target number of top combos
    const char* m       // Sorting mode ('avg' or 'min')
);

// Comparison functions for qsort and internal sorting
static int compare_avg_rank(const void* a, const void* b);
static int compare_min_rank(const void* a, const void* b);

// ----------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------
static void init_tables() {
    if (initialized) return;
    memset(nCk_table, 0, sizeof(nCk_table));
    for (int n = 0; n <= MAX_NUMBERS; n++) {
        nCk_table[n][0] = 1;
        for (int k_val = 1; k_val <= n && k_val <= MAX_K_FOR_SUBSET_EVAL; k_val++) {
            // Avoid overflow if C(n,k) gets huge, though uint64 should be sufficient for typical j,k
             if (nCk_table[n-1][k_val-1] > ULLONG_MAX - nCk_table[n-1][k_val]) {
                 nCk_table[n][k_val] = ULLONG_MAX; // Indicate overflow/too large
             } else {
                 nCk_table[n][k_val] = nCk_table[n-1][k_val-1] + nCk_table[n-1][k_val];
             }
        }
    }
    initialized = 1;
}

// ----------------------------------------------------------------------
// Popcount - use builtin if available
// ----------------------------------------------------------------------
static inline int popcount64(uint64 x) {
    #ifdef __GNUC__
        return __builtin_popcountll(x);
    #else
        // Fallback for non-GCC compilers (less efficient)
        int count = 0;
        while (x > 0) {
            x &= (x - 1);
            count++;
        }
        return count;
    #endif
}

// ----------------------------------------------------------------------
// Subset Hash Table (Linear Probing)
// ----------------------------------------------------------------------
static SubsetTable* create_subset_table() {
    SubsetTable* t = (SubsetTable*)malloc(sizeof(SubsetTable));
    if (!t) return NULL;
    // Use calloc for zero initialization (keys=0 means empty, values=-1 means empty)
    t->keys = (uint64*)calloc(HASH_SIZE, sizeof(uint64));
    t->values = (int*)malloc(HASH_SIZE * sizeof(int));
    if (!t->keys || !t->values) {
        free(t->keys);
        free(t->values);
        free(t);
        return NULL;
    }
    // Initialize values to -1 to distinguish from valid index 0
    for (int i = 0; i < HASH_SIZE; i++) {
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

// Simple hash function (using multiplication and XOR shift)
static inline uint32 hash_subset(uint64 pattern) {
    pattern = (pattern ^ (pattern >> 30)) * 0xbf58476d1ce4e5b9ULL;
    pattern = (pattern ^ (pattern >> 27)) * 0x94d049bb133111ebULL;
    pattern = pattern ^ (pattern >> 31);
    return (uint32)(pattern & (HASH_SIZE - 1));
}

// Insert pattern with its last seen index 'value'
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value) {
    uint32 idx = hash_subset(pattern);
    while (1) {
        // If slot is empty (key=0) or matches the pattern, insert/update
        if (table->keys[idx] == 0 || table->keys[idx] == pattern) {
            table->keys[idx] = pattern;
            table->values[idx] = value;
            return;
        }
        // Collision, move to next slot (linear probing)
        idx = (idx + 1) & (HASH_SIZE - 1);
        // Should ideally check for full table, but HASH_SIZE is large enough
    }
}

// Lookup pattern, return last seen index or -1 if not found
static inline int lookup_subset(const SubsetTable* table, uint64 pattern) {
    uint32 idx = hash_subset(pattern);
    while (1) {
        // If slot is empty (key=0), pattern not found
        if (table->keys[idx] == 0) return -1;
        // If keys match, return the value
        if (table->keys[idx] == pattern) return table->values[idx];
        // Collision, move to next slot
        idx = (idx + 1) & (HASH_SIZE - 1);
        // Should ideally check for wrap-around loop if table is full, but unlikely
    }
}

// ----------------------------------------------------------------------
// Combination / Draw Processing Helpers
// ----------------------------------------------------------------------
static inline uint64 numbers_to_pattern(const int* numbers, int count) {
    uint64 p = 0ULL;
    // Ensure numbers are within valid range (1 to 63 for uint64)
    for (int i = 0; i < count; i++) {
         if (numbers[i] >= 1 && numbers[i] <= 63) { // Check against 64 bit limit
             p |= (1ULL << (numbers[i] - 1));
         }
    }
    return p;
}

// Generate all k-subsets of a 6-number draw and insert into table
static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table) {
    // Check if k is valid for a 6-number draw and within precomputed C(n,k) range
    if (k <= 0 || k > 6 || k > MAX_K_FOR_SUBSET_EVAL) return;

    int subset_indices[MAX_K_FOR_SUBSET_EVAL]; // Indices into the draw array (0 to 5)
    for (int i = 0; i < k; i++) {
        subset_indices[i] = i;
    }

    while (1) {
        // Generate pattern for the current subset
        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) {
             int num = draw[subset_indices[i]];
             if (num >= 1 && num <= 63) {
                pat |= (1ULL << (num - 1));
             }
        }
        // Insert/update the pattern's last seen index
        insert_subset(table, pat, draw_idx);

        // Find the rightmost index to increment (Gosper's Hack equivalent for combinations)
        int pos_to_inc = k - 1;
        while (pos_to_inc >= 0 && subset_indices[pos_to_inc] == 6 - k + pos_to_inc) {
            pos_to_inc--;
        }

        // If all indices are maxed out, we're done
        if (pos_to_inc < 0) break;

        // Increment the found index
        subset_indices[pos_to_inc]++;

        // Reset subsequent indices
        for (int i = pos_to_inc + 1; i < k; i++) {
            subset_indices[i] = subset_indices[i - 1] + 1;
        }
    }
}


// ----------------------------------------------------------------------
// Backtracking Search with Optimized Pruning
// ----------------------------------------------------------------------
static void backtrack(
    // Current state
    int* S, int size, uint64 current_S,
    double current_min_rank, double sum_ranks_so_far, int num_subsets_so_far,
    int start_num,
    // Context
    const SubsetTable* table, int total_draws, int max_number, int j, int k, uint64 Cjk,
    // Top-L management
    ComboStats* thread_best, int* thread_filled, int l, const char* m
) {
    // --- Base Case: Combination complete ---
    if (size == j) {
        // Calculate final average rank
        double avg_rank = (Cjk > 0) ? (sum_ranks_so_far / (double)Cjk) : 0.0;
        double min_rank = current_min_rank;

        // --- Try to insert into thread's top-L list ---
        int should_insert = 0;
        if (*thread_filled < l) {
            // List not full, always insert
            should_insert = 1;
        } else {
            // List is full, compare with the *worst* element (at index l-1)
            ComboStats* worst_held = &thread_best[l - 1];
            if (strcmp(m, "avg") == 0) {
                // Higher avg_rank is better. If tied, higher min_rank is better.
                should_insert = (avg_rank > worst_held->avg_rank) ||
                                (avg_rank == worst_held->avg_rank && min_rank > worst_held->min_rank);
            } else { // m == "min"
                // Higher min_rank is better. If tied, higher avg_rank is better.
                should_insert = (min_rank > worst_held->min_rank) ||
                                (min_rank == worst_held->min_rank && avg_rank > worst_held->avg_rank);
            }
        }

        if (should_insert) {
            int insert_idx;
            if (*thread_filled < l) {
                // Append to end and increment count
                insert_idx = *thread_filled;
                *thread_filled = *thread_filled + 1;
            } else {
                // Replace the worst element
                insert_idx = l - 1;
            }

            // Store the new best combo's data
            thread_best[insert_idx].pattern = current_S;
            thread_best[insert_idx].avg_rank = avg_rank;
            thread_best[insert_idx].min_rank = min_rank;
            memcpy(thread_best[insert_idx].combo, S, j * sizeof(int));
            thread_best[insert_idx].len = j;

            // Bubble the newly inserted/updated element up to its correct sorted position
            // (Array is sorted descending by primary criterion)
            for (int i = insert_idx; i > 0; i--) {
                int should_swap = 0;
                ComboStats* current = &thread_best[i];
                ComboStats* prev = &thread_best[i - 1];
                if (strcmp(m, "avg") == 0) {
                    should_swap = (current->avg_rank > prev->avg_rank) ||
                                  (current->avg_rank == prev->avg_rank && current->min_rank > prev->min_rank);
                } else { // m == "min"
                    should_swap = (current->min_rank > prev->min_rank) ||
                                  (current->min_rank == prev->min_rank && current->avg_rank > prev->avg_rank);
                }

                if (should_swap) {
                    // Swap elements
                    ComboStats tmp = *current;
                    *current = *prev;
                    *prev = tmp;
                } else {
                    // Correct position found, stop bubbling
                    break;
                }
            }
        }
        return; // End recursion for this branch
    }

    // --- Pruning Check ---
    // Prune if the *best possible min_rank* down this path (current_min_rank)
    // is already worse than (less than) the min_rank of the worst combo
    // currently held in the top-L list. This applies to BOTH 'avg' and 'min' modes.
    if (*thread_filled == l && current_min_rank < thread_best[l - 1].min_rank) {
         // Optimization: If min_ranks are equal, we might still beat the worst combo
         // if the avg_rank turns out better (esp. in 'avg' mode).
         // So, only prune if strictly less.
         return; // Prune this branch
    }
    // Additional check: If we've already used up remaining slots for numbers
    if (max_number - start_num + 1 < j - size) {
        return; // Not enough numbers left to complete the combo
    }


    // --- Recursive Step: Try adding each possible next number ---
    for (int num = start_num; num <= max_number; num++) {
        // Add number 'num' to the current combination 'S'
        S[size] = num;
        uint64 new_S_pattern = current_S | (1ULL << (num - 1));

        // --- Evaluate newly formed k-subsets involving 'num' ---
        double min_rank_of_new_subsets = (double)total_draws; // Initialize to worst possible rank
        double sum_ranks_of_new_subsets = 0.0;
        int count_new_subsets = 0;

        // Check if enough elements exist to form k-1 subset from S[0...size-1]
        if (size >= k - 1 && k > 0 && k <= MAX_K_FOR_SUBSET_EVAL) {
            // Iterate through combinations of k-1 elements from S[0...size-1]
            int subset_indices[MAX_K_FOR_SUBSET_EVAL]; // Indices into S
            for (int i = 0; i < k - 1; i++) {
                subset_indices[i] = i;
            }

            while (1) {
                // Form the k-subset pattern including the new number 'num'
                uint64 subset_pat = (1ULL << (num - 1));
                for (int i = 0; i < k - 1; i++) {
                     int element_num = S[subset_indices[i]];
                     if (element_num >= 1 && element_num <= 63) {
                        subset_pat |= (1ULL << (element_num - 1));
                     }
                }

                // Lookup rank in the table
                int last_seen = lookup_subset(table, subset_pat);
                double rank = (last_seen >= 0) ? (double)(total_draws - last_seen - 1) : (double)total_draws;

                // Update stats for the new subsets
                sum_ranks_of_new_subsets += rank;
                if (rank < min_rank_of_new_subsets) {
                    min_rank_of_new_subsets = rank;
                }
                count_new_subsets++;

                // Find next combination of k-1 indices from S[0...size-1]
                int pos_to_inc = k - 2; // Rightmost index for k-1 elements
                // Find rightmost index that can be incremented
                while (pos_to_inc >= 0 && subset_indices[pos_to_inc] == size - (k - 1) + pos_to_inc) {
                    pos_to_inc--;
                }
                if (pos_to_inc < 0) break; // All combinations checked

                subset_indices[pos_to_inc]++;
                for (int i = pos_to_inc + 1; i < k - 1; i++) {
                    subset_indices[i] = subset_indices[i - 1] + 1;
                }
            }
        }

        // --- Update overall state for the recursive call ---
        double next_min_rank = (current_min_rank < min_rank_of_new_subsets) ? current_min_rank : min_rank_of_new_subsets;
        double next_sum_ranks = sum_ranks_so_far + sum_ranks_of_new_subsets;
        int next_num_subsets = num_subsets_so_far + count_new_subsets;

        // --- Recursive Call ---
        // Note: The pruning check happens *before* the loop and at the start of the next call
         backtrack(S, size + 1, new_S_pattern,
                   next_min_rank, next_sum_ranks, next_num_subsets,
                   num + 1, // Next number to try must be > current 'num'
                   table, total_draws, max_number, j, k, Cjk,
                   thread_best, thread_filled, l, m);
    }
}


// ----------------------------------------------------------------------
// Comparison Functions for Sorting ComboStats
// ----------------------------------------------------------------------
static int compare_avg_rank(const void* a, const void* b) {
    const ComboStats* ca = (const ComboStats*)a;
    const ComboStats* cb = (const ComboStats*)b;
    // Primary: Descending avg_rank
    if (ca->avg_rank > cb->avg_rank) return -1;
    if (ca->avg_rank < cb->avg_rank) return 1;
    // Secondary: Descending min_rank
    if (ca->min_rank > cb->min_rank) return -1;
    if (ca->min_rank < cb->min_rank) return 1;
    return 0;
}

static int compare_min_rank(const void* a, const void* b) {
    const ComboStats* ca = (const ComboStats*)a;
    const ComboStats* cb = (const ComboStats*)b;
    // Primary: Descending min_rank
    if (ca->min_rank > cb->min_rank) return -1;
    if (ca->min_rank < cb->min_rank) return 1;
    // Secondary: Descending avg_rank
    if (ca->avg_rank > cb->avg_rank) return -1;
    if (ca->avg_rank < cb->avg_rank) return 1;
    return 0;
}


// ----------------------------------------------------------------------
// Standard Analysis (Top-L with optional N non-overlapping)
// ----------------------------------------------------------------------
static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data, // All draws, sorted oldest to newest
    int use_count,           // Number of newest draws to consider
    int j, int k, const char* m, int l, int n,
    int max_number,
    int* out_len              // Output: number of results returned
) {
    *out_len = 0;
    // Basic validation
    if (use_count <= 0 || j <= 0 || k <= 0 || j < k || l <= 0 || n < 0 || k > MAX_K_FOR_SUBSET_EVAL || j > MAX_ALLOWED_J) {
        return NULL;
    }
     uint64 Cjk = nCk_table[j][k];
     if (Cjk == 0 || Cjk == ULLONG_MAX) { // Check if C(j,k) is valid/calculated
         fprintf(stderr, "Warning: C(%d, %d) is zero or too large.\n", j, k);
         return NULL;
     }

    // 1. Build the subset table using the relevant draws
    //    Draws are sorted_draws_data[0...draws_count-1]
    //    We use the last 'use_count' draws: indices [draws_count - use_count] to [draws_count - 1]
    //    Map these to indices 0 to use_count-1 for rank calculation.
    SubsetTable* table = create_subset_table();
    if (!table) return NULL;
    int start_idx_for_table = (int)fmax(0.0, (double)use_count - HASH_SIZE); // Optimization: Only hash relevant recent draws if table size is limited? No, hash all used draws.
    for (int i = 0; i < use_count; i++) {
        // Process draw original index 'draw_idx_orig' using its rank index 'i'
        process_draw(&sorted_draws_data[i * 6], i, k, table); // Pass index within the 'use_count' window
    }

    // 2. Prepare for parallel backtracking
    int num_threads = omp_get_max_threads();
    // Allocate space to store top-L results from *each* thread
    ComboStats* all_threads_best = (ComboStats*)malloc(num_threads * l * sizeof(ComboStats));
    int* all_threads_filled = (int*)calloc(num_threads, sizeof(int)); // Tracks filled count for each thread
    if (!all_threads_best || !all_threads_filled) {
        free(all_threads_best);
        free(all_threads_filled);
        free_subset_table(table);
        return NULL;
    }
     // Initialize ranks to a very small number to ensure they get replaced
    for(int i=0; i < num_threads * l; ++i) {
        all_threads_best[i].avg_rank = -DBL_MAX;
        all_threads_best[i].min_rank = -DBL_MAX;
        all_threads_best[i].len = 0;
    }


    int error_occurred = 0; // Flag for memory errors within parallel region

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        ComboStats* thread_best = &all_threads_best[thread_id * l];
        int* thread_filled = &all_threads_filled[thread_id];
        *thread_filled = 0; // Reset just in case

        int* S = (int*)malloc(j * sizeof(int));
        if (!S) {
            #pragma omp atomic write
            error_occurred = 1;
            // Ensure loop below doesn't run if allocation failed
             #pragma omp cancel parallel // Requires OMP >= 4.0, might not work everywhere
        }

        // Use dynamic scheduling as workload per first number can vary greatly
        #pragma omp for schedule(dynamic)
        for (int first = 1; first <= max_number - j + 1; first++) {
            // Check for cancellation or error before starting work
             if (error_occurred) continue; // Skip work if error occurred elsewhere


            S[0] = first;
            uint64 current_S = (1ULL << (first - 1));
            double current_min_rank = (double)use_count; // Initial best possible min rank is 'total_draws' (worst rank)
            double sum_ranks_so_far = 0.0;
            int num_subsets_so_far = 0;

            backtrack(S, 1, current_S,
                      current_min_rank, sum_ranks_so_far, num_subsets_so_far,
                      first + 1, // Start next number search from first + 1
                      table, use_count, max_number, j, k, Cjk,
                      thread_best, thread_filled, l, m);
        }

        if (S) free(S);
    } // End parallel region


    if (error_occurred) {
        fprintf(stderr, "Error: Memory allocation failed during parallel execution.\n");
        free(all_threads_best);
        free(all_threads_filled);
        free_subset_table(table);
        return NULL;
    }

    // 3. Merge results from all threads into a single candidate list
    int total_candidates_found = 0;
    for (int t = 0; t < num_threads; t++) {
        total_candidates_found += all_threads_filled[t];
    }

    if (total_candidates_found == 0) {
        free(all_threads_best);
        free(all_threads_filled);
        free_subset_table(table);
        return NULL; // No combinations found
    }

    ComboStats* final_candidates = (ComboStats*)malloc(total_candidates_found * sizeof(ComboStats));
    if (!final_candidates) {
        free(all_threads_best);
        free(all_threads_filled);
        free_subset_table(table);
        return NULL;
    }

    int current_candidate_idx = 0;
    for (int t = 0; t < num_threads; t++) {
        memcpy(&final_candidates[current_candidate_idx],
               &all_threads_best[t * l],
               all_threads_filled[t] * sizeof(ComboStats));
        current_candidate_idx += all_threads_filled[t];
    }

    free(all_threads_best); // No longer needed
    free(all_threads_filled);

    // 4. Sort the merged candidates according to 'm'
    if (strcmp(m, "avg") == 0) {
        qsort(final_candidates, total_candidates_found, sizeof(ComboStats), compare_avg_rank);
    } else {
        qsort(final_candidates, total_candidates_found, sizeof(ComboStats), compare_min_rank);
    }

    // 5. Select the actual top L results
    int top_l_count = (total_candidates_found < l) ? total_candidates_found : l;
    // Allocate space for final results (up to L for top + N for non-overlapping)
     AnalysisResultItem* results = (AnalysisResultItem*)calloc(l + n, sizeof(AnalysisResultItem));
     if (!results) {
         free(final_candidates);
         free_subset_table(table);
         return NULL;
     }
     int current_result_idx = 0;


    // 6. Format the top L results
    // Rebuild table if needed for formatting (or ensure table wasn't modified)
    // Table was read-only during backtrack, so it's still valid.
    for (int i = 0; i < top_l_count; i++) {
        AnalysisResultItem* res_item = &results[current_result_idx];
        ComboStats* src_stat = &final_candidates[i];

        format_combo(src_stat->combo, src_stat->len, res_item->combination);
        format_subsets(src_stat->combo, j, k, use_count, table, res_item->subsets);
        res_item->avg_rank = src_stat->avg_rank;
        res_item->min_value = src_stat->min_rank;
        res_item->is_chain_result = 0;
        // Fill other fields potentially needed by Python side (even if 0)
        res_item->draw_offset = 0;
        res_item->draws_until_common = 0;
        res_item->analysis_start_draw = 0;

        current_result_idx++;
    }

    // 7. Select up to N non-overlapping combinations (if n > 0)
    int selected_n_count = 0;
    if (n > 0 && top_l_count > 0) {
        // Keep track of patterns already chosen to provide non-overlapping results
        uint64* chosen_patterns = (uint64*)calloc(n, sizeof(uint64));
        int num_chosen = 0;

        // Iterate through the *sorted* top_l_count candidates
        for (int i = 0; i < top_l_count && num_chosen < n; i++) {
            uint64 current_pattern = final_candidates[i].pattern;
            int overlap_found = 0;

            // Check against already chosen patterns
            for (int chosen_idx = 0; chosen_idx < num_chosen; chosen_idx++) {
                uint64 intersection = current_pattern & chosen_patterns[chosen_idx];
                if (popcount64(intersection) >= k) {
                    overlap_found = 1;
                    break;
                }
            }

            if (!overlap_found) {
                // No overlap, select this one for the 'n' list
                AnalysisResultItem* res_item = &results[current_result_idx]; // Continue filling results array
                ComboStats* src_stat = &final_candidates[i];

                format_combo(src_stat->combo, src_stat->len, res_item->combination);
                format_subsets(src_stat->combo, j, k, use_count, table, res_item->subsets);
                res_item->avg_rank = src_stat->avg_rank;
                res_item->min_value = src_stat->min_rank;
                res_item->is_chain_result = 0; // Mark as non-chain result
                res_item->draw_offset = 0;
                res_item->draws_until_common = 0;
                res_item->analysis_start_draw = 0;

                chosen_patterns[num_chosen] = current_pattern;
                num_chosen++;
                current_result_idx++;
                selected_n_count++; // Count how many we added for the 'n' part
            }
        }
        free(chosen_patterns);
    }

    // 8. Cleanup and return
    free(final_candidates);
    free_subset_table(table);

    *out_len = current_result_idx; // Total number of items filled in results array
     if (*out_len == 0) {
         free(results); // Free if we didn't actually add anything
         return NULL;
     }
    return results;
}


// ----------------------------------------------------------------------
// Chain Analysis (Repeated Top-1)
// ----------------------------------------------------------------------
static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data, // All draws, sorted oldest to newest
    int draws_count,          // Total number of draws available
    int initial_offset,       // Starting offset (0 means use all draws initially)
    int j, int k, const char* m,
    int max_number,
    int* out_len              // Output: number of chain results
) {
    *out_len = 0;
    // Basic validation
    if (draws_count <= 0 || j <= 0 || k <= 0 || j < k || k > MAX_K_FOR_SUBSET_EVAL || j > MAX_ALLOWED_J || initial_offset >= draws_count) {
        return NULL;
    }
     uint64 Cjk = nCk_table[j][k];
     if (Cjk == 0 || Cjk == ULLONG_MAX) {
         fprintf(stderr, "Warning: C(%d, %d) is zero or too large for chain analysis.\n", j, k);
         return NULL;
     }


    // Allocate space for results - max possible chain length is draws_count
    // Use dynamic array or realloc if memory is a concern, but calloc is simpler for now.
    AnalysisResultItem* chain_results = (AnalysisResultItem*)calloc(draws_count, sizeof(AnalysisResultItem));
     if (!chain_results) {
         return NULL;
     }

    // Precompute bit patterns for all draws for faster overlap checking later
    uint64* draw_patterns = (uint64*)malloc(draws_count * sizeof(uint64));
    if (!draw_patterns) {
        free(chain_results);
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        draw_patterns[i] = numbers_to_pattern(&sorted_draws_data[i * 6], 6);
    }

    int current_offset = initial_offset;
    int chain_index = 0; // Index into chain_results array

    while (current_offset >= 0 && current_offset < draws_count) {
        int use_count = draws_count - current_offset;
        if (use_count <= 0) break; // No draws left to analyze

        // --- Find Top-1 Combo for current 'use_count' draws ---
        // Build subset table for these draws
        SubsetTable* table = create_subset_table();
        if (!table) break; // Error
         int start_idx_for_table = current_offset; // Use draws from current_offset to draws_count-1
         for (int i = 0; i < use_count; i++) {
             process_draw(&sorted_draws_data[(start_idx_for_table + i) * 6], i, k, table); // Index 0 to use_count-1
         }


        // Backtrack to find the single best combo (l=1)
        ComboStats best_combo_for_offset;
        int filled_count = 0;
        // Initialize best_combo_for_offset ranks to worst possible
        best_combo_for_offset.avg_rank = -DBL_MAX;
        best_combo_for_offset.min_rank = -DBL_MAX;
        best_combo_for_offset.len = 0;


        int* S = (int*)malloc(j * sizeof(int));
        if (!S) { free_subset_table(table); break; }

        // Since it's parallelized inside, just run the loop (though inefficient for L=1)
        // Or, run backtrack sequentially? Let's keep parallel for consistency.
        #pragma omp parallel
        {
            ComboStats thread_best; // Only need top 1 per thread
            int thread_filled = 0;
            int* S_thread = (int*)malloc(j * sizeof(int));
             // Initialize thread_best ranks
             thread_best.avg_rank = -DBL_MAX;
             thread_best.min_rank = -DBL_MAX;
             thread_best.len = 0;


            if(S_thread) {
                 #pragma omp for schedule(dynamic)
                 for (int first = 1; first <= max_number - j + 1; first++) {
                        S_thread[0] = first;
                        uint64 current_S_p = (1ULL << (first - 1));
                        double current_min_r = (double)use_count;
                        backtrack(S_thread, 1, current_S_p, current_min_r, 0.0, 0, first + 1,
                                  table, use_count, max_number, j, k, Cjk,
                                  &thread_best, &thread_filled, 1, m);
                 }

                 #pragma omp critical
                 {
                        if (thread_filled > 0) { // Did this thread find anything?
                            int is_better = 0;
                            if (filled_count == 0) {
                                is_better = 1;
                            } else {
                                if (strcmp(m, "avg") == 0) {
                                    is_better = (thread_best.avg_rank > best_combo_for_offset.avg_rank) ||
                                                (thread_best.avg_rank == best_combo_for_offset.avg_rank && thread_best.min_rank > best_combo_for_offset.min_rank);
                                } else { // min
                                    is_better = (thread_best.min_rank > best_combo_for_offset.min_rank) ||
                                                (thread_best.min_rank == best_combo_for_offset.min_rank && thread_best.avg_rank > best_combo_for_offset.avg_rank);
                                }
                            }
                            if (is_better) {
                                best_combo_for_offset = thread_best; // Copy struct
                                filled_count = 1;
                            }
                        }
                 }
                 free(S_thread);
            }
        } // End parallel region

         free(S); // Free outer scope S

        free_subset_table(table); // Done with table for this offset

        if (filled_count == 0) {
            // No combo found for this offset, chain ends
            break;
        }

        // --- Store the found Top-1 result ---
        AnalysisResultItem* res_item = &chain_results[chain_index];
        format_combo(best_combo_for_offset.combo, best_combo_for_offset.len, res_item->combination);
        // Regenerate subsets string using a temporary table for the current use_count
        {
             SubsetTable* temp_table = create_subset_table();
             if (temp_table) {
                int temp_start_idx = current_offset;
                for (int i = 0; i < use_count; i++) {
                    process_draw(&sorted_draws_data[(temp_start_idx + i) * 6], i, k, temp_table);
                }
                format_subsets(best_combo_for_offset.combo, j, k, use_count, temp_table, res_item->subsets);
                free_subset_table(temp_table);
             } else {
                 strcpy(res_item->subsets, "Error formatting subsets");
             }
        }

        res_item->avg_rank = best_combo_for_offset.avg_rank;
        res_item->min_value = best_combo_for_offset.min_rank;
        res_item->is_chain_result = 1;
        res_item->draw_offset = chain_index + 1; // Analysis # (1-based)
        res_item->analysis_start_draw = current_offset + 1; // Draw # where analysis started (1-based)

        // --- Find how many draws until a common subset appears ---
        uint64 combo_pattern = best_combo_for_offset.pattern;
        int draws_until_common = 0; // How many draws *after* the current analysis set start
        int next_offset_delta = 1; // Default jump if no common subset found

        // Check draws *before* the current analysis window (older draws)
        // We look from draw `current_offset - 1` down to 0.
        int common_found_at_draw = -1; // Original index of the draw
        for (int draw_idx_orig = current_offset - 1; draw_idx_orig >= 0; draw_idx_orig--) {
             uint64 prev_draw_pattern = draw_patterns[draw_idx_orig];
             uint64 intersection = combo_pattern & prev_draw_pattern;
             if (popcount64(intersection) >= k) {
                 common_found_at_draw = draw_idx_orig;
                 break; // Found the most recent common subset draw
             }
        }

        if (common_found_at_draw != -1) {
            // Found common subset at original index 'common_found_at_draw'
            // Draws until common is the difference in original indices
            draws_until_common = (current_offset - 1) - common_found_at_draw;
            next_offset_delta = draws_until_common + 1;
        } else {
            // No common subset found in past draws. Assume it occurs right after the last past draw.
            // The duration is effectively all past draws.
            draws_until_common = current_offset; // All draws before the current window
            next_offset_delta = current_offset + 1; // Jump past all previous draws
        }

        res_item->draws_until_common = draws_until_common;

        // --- Update offset for next iteration ---
        current_offset -= next_offset_delta;
        chain_index++;

        if(chain_index >= draws_count) break; // Safety check

    } // End while loop for chain


    free(draw_patterns);
    *out_len = chain_index;

    if (chain_index == 0) {
        free(chain_results);
        return NULL; // No results generated
    }

    // Optionally resize chain_results if using realloc, otherwise return as is.
    return chain_results;
}


// ----------------------------------------------------------------------
// Public API Functions (called by Python via ctypes)
// ----------------------------------------------------------------------

/**
 * run_analysis_c(...)
 * Main entry point from Python.
 */
AnalysisResultItem* run_analysis_c(
    const char* game_type,    // e.g., "6_42" or "6_49"
    int** draws,              // Array of pointers to draw arrays (int[6])
    int draws_count,          // Total number of draws provided
    int j, int k, const char* m, // Analysis parameters
    int l, int n,             // Top-L, Non-overlapping-N
    int last_offset,          // Offset from the *last* draw (0 means use all draws)
    int* out_len              // Output: number of results returned
) {
    *out_len = 0;
    // Basic input validation
    if (!game_type || !draws || draws_count <= 0 || j <= 0 || k <= 0 || k > j || !m || l == 0 || n < 0 || last_offset < 0 || last_offset >= draws_count) {
        // Allow l = -1 for chain analysis
        if (l != -1 || (l == -1 && (draws_count <= 0 || j <= 0 || k <= 0 || k > j))) {
            fprintf(stderr, "Error: Invalid parameters passed to run_analysis_c.\n");
            return NULL;
        }
    }
    if (k > MAX_K_FOR_SUBSET_EVAL) {
         fprintf(stderr, "Error: Parameter k=%d exceeds maximum supported k=%d\n", k, MAX_K_FOR_SUBSET_EVAL);
         return NULL;
    }
     if (j > MAX_ALLOWED_J) {
         fprintf(stderr, "Error: Parameter j=%d exceeds maximum supported j=%d\n", j, MAX_ALLOWED_J);
         return NULL;
    }


    init_tables(); // Ensure combinations table is ready

    // Determine max number based on game type
    int max_number = (strstr(game_type, "6_49")) ? 49 : 42; // Default to 6/42 if not 6/49

    // --- Prepare draws data ---
    // The input `draws` are assumed to be sorted by draw order (oldest first).
    // We need a contiguous block of memory holding the numbers,
    // with each draw's 6 numbers internally sorted ascending.
    int* sorted_draws_data = (int*)malloc(draws_count * 6 * sizeof(int));
    if (!sorted_draws_data) {
        fprintf(stderr, "Error: Failed to allocate memory for draws data.\n");
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        int temp[6];
        // Check if input pointer is valid
        if (!draws[i]) {
            fprintf(stderr, "Error: Invalid draw pointer encountered at index %d.\n", i);
            free(sorted_draws_data);
            return NULL;
        }
        memcpy(temp, draws[i], 6 * sizeof(int));

        // Simple bubble sort for the 6 numbers within a draw
        for (int a = 0; a < 5; a++) {
            for (int b = a + 1; b < 6; b++) {
                if (temp[a] > temp[b]) {
                    int t = temp[a]; temp[a] = temp[b]; temp[b] = t;
                }
            }
        }
        // Copy sorted numbers to the contiguous block
        memcpy(&sorted_draws_data[i * 6], temp, 6 * sizeof(int));
    }


    // --- Dispatch to appropriate analysis function ---
    AnalysisResultItem* results = NULL;
    if (l != -1) {
        // Standard analysis (Top-L, N non-overlapping)
        int use_count = draws_count - last_offset;
        if (use_count > 0) {
             // Pass pointer to the start of the relevant data for the subset table
             results = run_standard_analysis(
                 &sorted_draws_data[last_offset * 6], // Pointer to the first draw to use
                 use_count,
                 j, k, m, l, n, max_number,
                 out_len
             );
        } else {
            *out_len = 0; // Ensure out_len is 0 if use_count is invalid
        }
    } else {
        // Chain analysis (l == -1)
        // run_chain_analysis expects all draws and the initial offset
        results = run_chain_analysis(
            sorted_draws_data, // Pass all draws
            draws_count,
            last_offset,       // The starting offset
            j, k, m,
            max_number,
            out_len
        );
    }

    // --- Cleanup and return ---
    free(sorted_draws_data); // Free the contiguous draws data block
    return results; // Return the results (or NULL if error/no results)
}

/**
 * free_analysis_results(...)
 * Frees the memory allocated for the results array by run_analysis_c.
 */
void free_analysis_results(AnalysisResultItem* results) {
    if (results) {
        free(results);
    }
}


// ----------------------------------------------------------------------
// Formatting Helpers (Identical to original/current provided code)
// ----------------------------------------------------------------------
static void format_combo(const int* combo, int len, char* out) {
    int pos = 0;
    for (int i = 0; i < len; i++) {
        if (pos >= MAX_COMBO_STR - 10) break; // Prevent buffer overflow
        if (i > 0) {
            out[pos++] = ',';
            out[pos++] = ' ';
        }
        pos += snprintf(out + pos, MAX_COMBO_STR - pos, "%d", combo[i]);
    }
    out[pos] = '\0';
}

// Helper struct for sorting subsets by rank for formatting
typedef struct {
    int numbers[MAX_K_FOR_SUBSET_EVAL];
    int rank;
} SubsetInfo;

static int compare_subset_rank(const void* a, const void* b) {
    const SubsetInfo* sa = (const SubsetInfo*)a;
    const SubsetInfo* sb = (const SubsetInfo*)b;
    // Descending rank
    if (sa->rank > sb->rank) return -1;
    if (sa->rank < sb->rank) return 1;
    // Ascending subset numbers for tie-breaking (arbitrary but consistent)
    for (int i = 0; i < MAX_K_FOR_SUBSET_EVAL; i++) {
        if (sa->numbers[i] < sb->numbers[i]) return -1;
        if (sa->numbers[i] > sb->numbers[i]) return 1;
    }
    return 0;
}


static void format_subsets(const int* combo, int j, int k, int total_draws,
                          const SubsetTable* table, char* out)
{
    // Check bounds
    if (k <= 0 || k > j || k > MAX_K_FOR_SUBSET_EVAL || j > MAX_ALLOWED_J) {
        strcpy(out, "[]");
        return;
    }

    uint64 exact_subset_count_u64 = nCk_table[j][k];
     if (exact_subset_count_u64 == 0 || exact_subset_count_u64 == ULLONG_MAX || exact_subset_count_u64 > INT_MAX) {
         strcpy(out, "[]"); // Too many subsets to handle reasonably
         return;
     }
     int exact_subset_count = (int)exact_subset_count_u64;


    SubsetInfo* subsets_info = (SubsetInfo*)malloc(exact_subset_count * sizeof(SubsetInfo));
    if (!subsets_info) {
        strcpy(out, "[]"); // Allocation failed
        return;
    }
    int subset_count = 0; // Actual count added

    // Iterate through k-subsets of the combo
    int subset_indices[MAX_K_FOR_SUBSET_EVAL]; // Indices into the combo array
    for (int i = 0; i < k; i++) {
        subset_indices[i] = i;
    }

    while (subset_count < exact_subset_count) {
        // Form pattern and numbers array for current subset
        uint64 pat = 0ULL;
        SubsetInfo current_info;
        memset(&current_info, 0, sizeof(SubsetInfo)); // Clear numbers

        for (int i = 0; i < k; i++) {
            int num = combo[subset_indices[i]];
             current_info.numbers[i] = num;
             if (num >= 1 && num <= 63) {
                pat |= (1ULL << (num - 1));
             }
        }

        // Lookup rank
        int last_seen = lookup_subset(table, pat);
        current_info.rank = (last_seen >= 0) ? (total_draws - last_seen - 1) : total_draws;

        // Store info
        subsets_info[subset_count++] = current_info;

        // Next combination indices
        int p = k - 1;
        while (p >= 0 && subset_indices[p] == j - k + p) p--;
        if (p < 0) break; // Done
        subset_indices[p]++;
        for (int x = p + 1; x < k; x++) {
            subset_indices[x] = subset_indices[x - 1] + 1;
        }
    }

    // Sort the found subsets by rank (descending)
    qsort(subsets_info, subset_count, sizeof(SubsetInfo), compare_subset_rank);

    // Format the output string
    int pos = 0;
    out[pos++] = '[';
    int remaining_space = MAX_SUBSETS_STR - 2; // Account for '[' and ']' and '\0'

    for (int i = 0; i < subset_count && remaining_space > 1; i++) {
        if (i > 0) {
            if (remaining_space < 2) break;
            out[pos++] = ',';
            out[pos++] = ' ';
            remaining_space -= 2;
        }

        char subset_buf[256]; // Buffer for a single subset entry
        int current_len = 0;

        // Format subset numbers: ((num1, num2, ...), rank)
        current_len += snprintf(subset_buf + current_len, sizeof(subset_buf) - current_len, "((");
        for (int n = 0; n < k; n++) {
             if (current_len >= sizeof(subset_buf) - 10) break; // Prevent overflow
             current_len += snprintf(subset_buf + current_len, sizeof(subset_buf) - current_len,
                                     "%s%d", (n > 0) ? ", " : "", subsets_info[i].numbers[n]);
        }
         if (current_len < sizeof(subset_buf) - 10) {
             current_len += snprintf(subset_buf + current_len, sizeof(subset_buf) - current_len,
                                     "), %d)", subsets_info[i].rank);
         } else {
             // Truncate if too long
             strcpy(subset_buf + sizeof(subset_buf) - 5, "...)");
             current_len = strlen(subset_buf);
         }


        if (current_len >= remaining_space) {
            // Not enough space for this subset entry, possibly add ellipsis if desired
             if (remaining_space > 3) {
                 out[pos++] = '.'; out[pos++] = '.'; out[pos++] = '.'; remaining_space -= 3;
             }
            break;
        }

        memcpy(out + pos, subset_buf, current_len);
        pos += current_len;
        remaining_space -= current_len;
    }

    out[pos++] = ']';
    out[pos] = '\0';

    free(subsets_info);
}
