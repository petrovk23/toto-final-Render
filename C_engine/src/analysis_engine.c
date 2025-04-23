#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <float.h> // For DBL_MAX, DBL_MIN
#include <stdint.h> // For UINT64_MAX

#include "analysis_engine.h"

// ----------------------------------------------------------------------
// Defines and Typedefs
// ----------------------------------------------------------------------
#define MAX_COMBO_STR 255        // Max length for formatted combo string
#define MAX_SUBSETS_STR 65535    // Max length for formatted subsets string
#define MAX_ALLOWED_J 200        // Max allowed size of combination (j)
#define MAX_NUMBERS 50           // Max number expected in draws (e.g., 49 for 6/49) - Affects nCk table size
#define MAX_DRAW_SIZE 6          // Fixed number of integers per draw
#define HASH_SIZE (1 << 26)      // Size of the subset hash table (power of 2) - 67M entries

typedef unsigned long long uint64;
typedef unsigned int uint32;

// ----------------------------------------------------------------------
// Global Variables and Static Data
// ----------------------------------------------------------------------
// Precomputed nCk table (Pascal's triangle)
static uint64 nCk_table[MAX_NUMBERS + 1][MAX_NUMBERS + 1]; // Size based on MAX_NUMBERS
static int initialized = 0; // Flag to ensure tables are initialized only once

// Structure to store subset pattern -> last seen draw index mapping (hash table)
typedef struct {
    uint64* keys;     // Array of subset bit patterns (keys)
    int* values;      // Array of last seen draw indices (values), -1 for empty
    int size;         // Current number of entries (unused in current implementation)
    int capacity;     // Capacity of the keys/values arrays (HASH_SIZE)
} SubsetTable;

// Structure to hold statistics for a single j-combination during processing
typedef struct {
    uint64 pattern;   // Bit pattern of the combo's numbers (1ULL << (num - 1))
    double avg_rank;  // Average rank (score) of its k-subsets
    double min_rank;  // Minimum rank (score) among its k-subsets
    int combo[MAX_ALLOWED_J]; // The numbers in the combo (up to MAX_ALLOWED_J)
    int len;          // Length of the combo (should be 'j' when valid)
} ComboStats;

// ----------------------------------------------------------------------
// Forward Declarations of Static Functions
// ----------------------------------------------------------------------
static void init_tables();
static inline int popcount64(uint64 x); // Count set bits in a 64-bit integer
static SubsetTable* create_subset_table(int max_entries);
static void free_subset_table(SubsetTable* table);
static inline uint32 hash_subset(uint64 pattern); // Hash function for subset patterns
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value); // Insert pattern->draw_index
static inline int lookup_subset(const SubsetTable* table, uint64 pattern); // Lookup pattern->draw_index
static inline uint64 numbers_to_pattern(const int* numbers, int count); // Convert numbers array to bit pattern
static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table); // Process k-subsets of a draw
static void format_combo(const int* combo, int len, char* out); // Format combo numbers to string
static void format_subsets(const int* combo, int j, int k, int total_draws,
                           const SubsetTable* table, char* out); // Format sorted subsets list to string

// Main analysis routine dispatchers
static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data, int use_count,
    int j, int k, const char* m, int l, int n,
    int max_number, int* out_len
);
static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data, int draws_count, int initial_offset,
    int j, int k, const char* m,
    int max_number, int* out_len
);

// Helper function for inserting into top-L list
static int insert_into_best(ComboStats* best_list, int l, int* filled_count,
                            double new_avg, double new_min, const int* new_combo, int j, uint64 new_pattern,
                            const char* m);

// Core backtracking function for finding top-L combinations
static void backtrack(
    int* S, int size, uint64 current_S,
    double current_min_rank, double sum_current,
    int start_num, SubsetTable* table, int total_draws, int max_number,
    int j, int k, ComboStats* thread_best, int* thread_filled, int l,
    const char* m, uint64 Cjk
);

// ----------------------------------------------------------------------
// Initialization Function
// ----------------------------------------------------------------------
/**
 * @brief Initializes global lookup tables (nCk). Called once.
 */
static void init_tables() {
    if (initialized) return;
    memset(nCk_table, 0, sizeof(nCk_table));
    for (int n = 0; n <= MAX_NUMBERS; n++) {
        nCk_table[n][0] = 1; // C(n, 0) = 1
        for (int k_ = 1; k_ <= n; k_++) {
            // C(n, k) = C(n-1, k-1) + C(n-1, k)
            // Check for potential overflow if MAX_NUMBERS is large, though uint64 is capacious.
            uint64 term1 = nCk_table[n-1][k_-1];
            uint64 term2 = nCk_table[n-1][k_];
             // Basic check: if sum would be less than one term, overflow occurred.
            if (term1 > UINT64_MAX - term2) {
                 // Handle overflow - e.g., saturate or set to 0/error indicator
                 // For MAX_NUMBERS=50, C(50, 25) fits in uint64. This check might be unnecessary.
                 nCk_table[n][k_] = UINT64_MAX; // Saturate on overflow
            } else {
                nCk_table[n][k_] = term1 + term2;
            }
        }
    }
    initialized = 1;
}

// ----------------------------------------------------------------------
// Utility Functions (Inline for Performance)
// ----------------------------------------------------------------------
/**
 * @brief Counts the number of set bits (1s) in a 64-bit integer.
 * Uses GCC/Clang built-in intrinsic for efficiency.
 */
static inline int popcount64(uint64 x) {
    #ifdef __GNUC__
    return __builtin_popcountll(x);
    #else
    // Fallback implementation if not using GCC/Clang (less efficient)
    int count = 0;
    while (x > 0) {
        x &= (x - 1); // Clear the least significant bit set
        count++;
    }
    return count;
    #endif
}

/**
 * @brief Creates and initializes a hash table for subset storage.
 * @param max_entries The desired capacity (should be HASH_SIZE).
 * @return Pointer to the created table, or NULL on allocation failure.
 */
static SubsetTable* create_subset_table(int max_entries) {
    SubsetTable* t = (SubsetTable*)malloc(sizeof(SubsetTable));
    if (!t) return NULL;
    t->size = 0;
    t->capacity = max_entries;
    // Use calloc for keys to initialize to 0 (useful if 0 is not a valid pattern)
    t->keys = (uint64*)calloc(max_entries, sizeof(uint64));
    // Initialize values to -1 to reliably signify empty slots
    t->values = (int*)malloc(max_entries * sizeof(int));
    if (!t->keys || !t->values) {
        free(t->keys); // free(NULL) is safe
        free(t->values);
        free(t);
        return NULL;
    }
    for (int i = 0; i < max_entries; i++) {
        t->values[i] = -1; // -1 indicates empty slot
    }
    return t;
}

/**
 * @brief Frees the memory associated with a SubsetTable.
 */
static void free_subset_table(SubsetTable* table) {
    if (!table) return;
    free(table->keys);
    free(table->values);
    free(table);
}

/**
 * @brief Simple multiplicative hash function for 64-bit subset patterns.
 * @param pattern The 64-bit subset pattern.
 * @return A 32-bit hash index within the table bounds [0, HASH_SIZE - 1].
 */
static inline uint32 hash_subset(uint64 pattern) {
    // Using a simple mixing sequence (similar to FNV or MurmurHash components)
    const uint64 prime = 0x100000001b3ULL; // FNV prime
    uint64 hash = 0xcbf29ce484222325ULL;  // FNV offset basis
    hash ^= pattern;
    hash *= prime;
    hash ^= hash >> 32; // Mix high bits
    return (uint32)(hash & (HASH_SIZE - 1)); // Mask to fit table size (must be power of 2)
}


/**
 * @brief Inserts a subset pattern and its last seen draw index into the hash table.
 * Uses linear probing to resolve collisions. Updates value if key already exists.
 * @param table The subset table.
 * @param pattern The subset pattern (key).
 * @param value The draw index (value).
 */
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value) {
    uint32 idx = hash_subset(pattern);
    uint32 start_idx = idx; // To detect full table loop
    while (1) {
        // If slot is empty (value == -1) or contains the same key, insert/update
        if (table->values[idx] == -1 || table->keys[idx] == pattern) {
            table->keys[idx] = pattern;
            table->values[idx] = value; // Update with the latest draw index
            // Optionally increment table->size if tracking accurately
            return;
        }
        // Collision, move to next slot (linear probing with wrap-around)
        idx = (idx + 1) & (HASH_SIZE - 1);
         // Safety check: If we've probed the entire table and returned to start, it's full.
        if (idx == start_idx) {
             // This should ideally not happen if HASH_SIZE is large enough.
             // Handle error: print message, exit, or just stop inserting?
             fprintf(stderr, "Warning: Subset hash table full or looping. Stopping insertion.\n");
             return;
        }
    }
}

/**
 * @brief Looks up a subset pattern in the hash table.
 * @param table The subset table.
 * @param pattern The subset pattern to find.
 * @return The last seen draw index if found, or -1 if not found.
 */
static inline int lookup_subset(const SubsetTable* table, uint64 pattern) {
    uint32 idx = hash_subset(pattern);
    uint32 start_idx = idx; // To detect loop if key not found in full table
    while (1) {
        // If slot is empty, pattern not found
        if (table->values[idx] == -1) return -1;
        // If key matches, return value
        if (table->keys[idx] == pattern) return table->values[idx];
        // Collision, move to next slot
        idx = (idx + 1) & (HASH_SIZE - 1);
        // Safety check: If we've probed the entire table and returned, key not present.
        if (idx == start_idx) return -1;
    }
}

/**
 * @brief Converts an array of numbers into a 64-bit bitmask pattern.
 * Assumes numbers are positive and within the range [1, 64].
 * @param numbers Pointer to the array of numbers.
 * @param count The number of elements in the array.
 * @return The corresponding bitmask pattern.
 */
static inline uint64 numbers_to_pattern(const int* numbers, int count) {
    uint64 p = 0ULL;
    for (int i = 0; i < count; i++) {
        // Ensure numbers are within valid range [1, 64] for bitmask
        if (numbers[i] > 0 && numbers[i] <= 64) {
            p |= (1ULL << (numbers[i] - 1));
        }
    }
    return p;
}

/**
 * @brief Processes a single draw: generates all its k-subsets and inserts/updates them in the table.
 * Uses a standard algorithm to iterate through combinations C(draw_size, k).
 * @param draw Pointer to the start of the sorted draw numbers (size MAX_DRAW_SIZE).
 * @param draw_idx The index of this draw relative to the analysis window start (0-based).
 * @param k The size of subsets to generate.
 * @param table The subset hash table.
 */
static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table) {
    if (k <= 0 || k > MAX_DRAW_SIZE) return; // Basic validation

    int combo_indices[MAX_DRAW_SIZE]; // Indices into the 'draw' array [0...MAX_DRAW_SIZE-1]
    for (int i = 0; i < k; i++) combo_indices[i] = i; // Initialize to first k indices

    while (1) {
        // Generate subset pattern from current indices
        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) {
             int num = draw[combo_indices[i]];
             if (num > 0 && num <= 64) { // Check bounds before setting bit
                pat |= (1ULL << (num - 1));
             }
        }
        // Insert/update the pattern with the current draw index
        insert_subset(table, pat, draw_idx);

        // Generate next combination of indices (standard algorithm for C(n, k))
        // Here n = MAX_DRAW_SIZE (e.g., 6)
        int p = k - 1;
        // Find rightmost index that can be incremented
        while (p >= 0 && combo_indices[p] == MAX_DRAW_SIZE - k + p) {
            p--;
        }
        // If no index can be incremented (p < 0), we are done iterating combinations
        if (p < 0) break;

        // Increment the found index
        combo_indices[p]++;

        // Reset subsequent indices to the smallest possible values
        for (int x = p + 1; x < k; x++) {
            combo_indices[x] = combo_indices[x - 1] + 1;
        }
    }
}

// ----------------------------------------------------------------------
// Helper for Managing Top-L Lists
// ----------------------------------------------------------------------
/**
 * @brief Inserts a new combination into a top-L list if it qualifies.
 * The list is sorted descending based on the primary key ('m') and secondary key.
 * Higher scores (ranks) are considered better.
 * @param best_list The array holding the current top-L ComboStats.
 * @param l The capacity of the list (L).
 * @param filled_count Pointer to the current number of items in the list.
 * @param new_avg Average rank (score) of the new combo.
 * @param new_min Minimum rank (score) of the new combo.
 * @param new_combo Pointer to the numbers of the new combo.
 * @param j The size of the new combo.
 * @param new_pattern Bit pattern of the new combo.
 * @param m Sorting mode ("avg" or "min").
 * @return 1 if the item was inserted, 0 otherwise. Updates *filled_count.
 */
static int insert_into_best(ComboStats* best_list, int l, int* filled_count,
                            double new_avg, double new_min, const int* new_combo, int j, uint64 new_pattern,
                            const char* m)
{
    // Determine primary and secondary scores for the new item based on mode 'm'
    double primary_new = (strcmp(m, "avg") == 0) ? new_avg : new_min;
    double secondary_new = (strcmp(m, "avg") == 0) ? new_min : new_avg;

    // If the list is full, check if the new item is better than the worst item currently in the list
    if (*filled_count == l) {
        double primary_worst = (strcmp(m, "avg") == 0) ? best_list[l - 1].avg_rank : best_list[l - 1].min_rank;
        double secondary_worst = (strcmp(m, "avg") == 0) ? best_list[l - 1].min_rank : best_list[l - 1].avg_rank;

        // If new item is not better than the current worst (lower primary, or equal primary and lower/equal secondary), return
        if (primary_new < primary_worst || (primary_new == primary_worst && secondary_new <= secondary_worst)) {
            return 0; // Not good enough to replace the worst
        }
    }

    // Find the correct insertion index 'idx' where the new item should go to maintain sorted order (desc)
    int idx = l; // Start by assuming it doesn't fit or goes last if replacing
    int current_list_size = (*filled_count < l) ? *filled_count : l; // Number of items to compare against
    for (int k = 0; k < current_list_size; ++k) {
        double primary_k = (strcmp(m, "avg") == 0) ? best_list[k].avg_rank : best_list[k].min_rank;
        double secondary_k = (strcmp(m, "avg") == 0) ? best_list[k].min_rank : best_list[k].avg_rank;

        // If new item is better than item at index k (higher primary, or equal primary and higher secondary)
        if (primary_new > primary_k || (primary_new == primary_k && secondary_new > secondary_k)) {
            idx = k; // Found the insertion point (will insert *at* this index)
            break;
        }
    }

    // If idx < l, the new item belongs in the top-L list (either inserting or replacing)
    if (idx < l) {
        // Shift elements down from idx to make space for the new item
        // Determine the starting point for shifting (end of current filled list or end of array if full)
        int shift_start = (*filled_count < l) ? *filled_count : l - 1;
        for (int move_idx = shift_start; move_idx > idx; --move_idx) {
             // Check bounds: only copy if source (move_idx-1) and dest (move_idx) are valid
             if (move_idx > 0 && move_idx < l) { // Ensure move_idx is within [1, l-1]
                 memcpy(&best_list[move_idx], &best_list[move_idx - 1], sizeof(ComboStats));
             }
        }

        // Insert the new element data at the found index 'idx'
        memcpy(best_list[idx].combo, new_combo, j * sizeof(int));
        best_list[idx].len = j;
        best_list[idx].avg_rank = new_avg;
        best_list[idx].min_rank = new_min;
        best_list[idx].pattern = new_pattern;

        // Increment filled count if the list wasn't full before insertion
        if (*filled_count < l) {
            (*filled_count)++;
        }
        return 1; // Item was successfully inserted
    }

    return 0; // Item did not make it into the top L
}

// ----------------------------------------------------------------------
// Core Backtracking Algorithm
// ----------------------------------------------------------------------
/**
 * @brief Recursively explores j-combinations to find the top-L based on avg/min rank.
 * Uses pruning based on bounds derived from partial combinations.
 */
static void backtrack(
    int* S,                 // Current partial combination being built
    int size,               // Current size of S (number of elements chosen)
    uint64 current_S,       // Bitmask pattern of S
    double current_min_rank,// Minimum rank (score) found among k-subsets involving S[0...size-1]
    double sum_current,     // Sum of ranks (scores) for k-subsets involving S[0...size-1]
    int start_num,          // Next number to consider adding (ensures ascending order)
    SubsetTable* table,     // Hash table for subset rank lookups
    int total_draws,        // Number of draws being analyzed (use_count or chain window size)
    int max_number,         // Max possible number in a draw (e.g., 49)
    int j,                  // Target size of combination
    int k,                  // Size of subsets
    ComboStats* thread_best,// Array to store top-L combos for the current thread
    int* thread_filled,     // Pointer to number of combos currently in thread_best
    int l,                  // Number of top combos to find (L)
    const char* m,          // Sorting mode ("avg" or "min")
    uint64 Cjk              // Precomputed C(j, k): total number of k-subsets in a full j-combination
) {
    // === Base Case: Combination is complete (size == j) ===
    if (size == j) {
        // Calculate final average rank (score) for the completed combination
        // Note: sum_current now holds the sum for all C(j,k) subsets.
        // Note: current_min_rank now holds the minimum rank for all C(j,k) subsets.
        double avg_rank = (Cjk > 0) ? (sum_current / (double)Cjk) : 0.0; // Avoid division by zero if Cjk=0
        double min_rank = current_min_rank;

        // Attempt to insert this completed combination into the thread's top-L list
        insert_into_best(thread_best, l, thread_filled, avg_rank, min_rank, S, j, current_S, m);
        return; // End recursion for this branch
    }

    // === Pruning Threshold Calculation ===
    // Only calculate threshold if the thread's list is full (allows pruning)
    double threshold_rank = -DBL_MAX; // Initialize to worst possible score (higher is better)
    double threshold_secondary_rank = -DBL_MAX;
    if (*thread_filled == l) {
        // Get the primary and secondary scores of the current worst combo in the list
        if (strcmp(m, "avg") == 0) {
            threshold_rank = thread_best[l - 1].avg_rank;
            threshold_secondary_rank = thread_best[l - 1].min_rank;
        } else { // "min" mode
            threshold_rank = thread_best[l - 1].min_rank;
            threshold_secondary_rank = thread_best[l - 1].avg_rank;
        }
    }

    // === Recursive Step: Try adding the next number ===
    // Iterate through possible numbers 'num' to add at index 'size'
    // Calculate upper bound for 'num' to ensure enough remaining numbers can form a j-combo
    int upper_bound_num = max_number - (j - (size + 1));
    for (int num = start_num; num <= upper_bound_num; num++) {

        // Add number 'num' to the current partial combination S
        S[size] = num;
        // Avoid bitwise ops if num > 64, although checks elsewhere should prevent this.
        uint64 bit_for_num = (num > 0 && num <= 64) ? (1ULL << (num - 1)) : 0;
        uint64 new_S = current_S | bit_for_num;

        // --- Calculate rank contributions of *new* k-subsets formed by adding 'num' ---
        // These are subsets containing 'num' and k-1 elements from S[0...size-1]
        double min_of_new_subsets = DBL_MAX; // Initialize to a value higher than any possible rank
        double sum_of_new_subsets = 0.0;

        // Only need to calculate if we have enough previous elements (size >= k-1)
        uint64 num_new_subsets_expected = 0; // C(size, k-1)
        if (size >= k - 1 && k > 0) {
             // Check if C(size, k-1) fits in uint64 and is non-zero
             num_new_subsets_expected = (size >= k-1 && k-1 >= 0) ? nCk_table[size][k-1] : 0;

             if (num_new_subsets_expected > 0) {
                 min_of_new_subsets = (double)total_draws + 1.0; // Reset for actual calculation

                 int combo_indices[k - 1]; // Indices into S[0...size-1]
                 for (int i = 0; i < k - 1; ++i) combo_indices[i] = i; // Start with first k-1 elements of S

                 while (1) {
                    // Form the k-subset pattern including 'num' and elements from S at combo_indices
                    uint64 pat = bit_for_num; // Start pattern with the new number's bit
                    for (int i = 0; i < k - 1; ++i) {
                        int prev_num = S[combo_indices[i]];
                        if (prev_num > 0 && prev_num <= 64) { // Check bounds
                            pat |= (1ULL << (prev_num - 1));
                        }
                    }

                    // Lookup rank (score) from hash table
                    int last_seen = lookup_subset(table, pat);
                    // Rank = draws ago (0 = most recent), total_draws = never seen. Higher is better score.
                    double rank = (last_seen >= 0) ? (double)(total_draws - last_seen - 1) : (double)total_draws;

                    // Update statistics for the new subsets found in this step
                    if (rank < min_of_new_subsets) min_of_new_subsets = rank;
                    sum_of_new_subsets += rank;

                    // Find next combination of (k-1) indices from S[0...size-1]
                    int p = k - 2; // Index runs from 0 to k-2 for k-1 elements
                    // Find rightmost index that can be incremented (relative to 'size' elements available)
                    while (p >= 0 && combo_indices[p] == size - (k - 1 - p)) {
                        p--;
                    }
                    if (p < 0) break; // No more combinations of k-1 from size

                    combo_indices[p]++;
                    for (int x = p + 1; x < k - 1; ++x) {
                        combo_indices[x] = combo_indices[x - 1] + 1;
                    }
                 } // End while loop for new subset combinations
             } // End if num_new_subsets_expected > 0
        } // End if (size >= k - 1)

        // Update overall stats for the extended partial combo S[0...size]
        // next_min_rank is the min over all subsets evaluated up to this point
        double next_min_rank = (current_min_rank < min_of_new_subsets) ? current_min_rank : min_of_new_subsets;
        // next_sum_current is the sum over all subsets evaluated up to this point
        double next_sum_current = sum_current + sum_of_new_subsets;

        // --- Pruning Check ---
        // Determine if we should continue exploring this branch
        int should_continue = 1; // Default: continue recursion
        if (*thread_filled == l) { // Only apply pruning if the top-L list is full

            if (strcmp(m, "min") == 0) {
                // --- Pruning for 'min' mode ---
                // Primary key: min_rank (higher is better). Secondary: avg_rank (higher is better).
                // Prune if the current minimum rank ('next_min_rank') can *never* beat the threshold.
                // Calculate the best possible average rank from here (assuming remaining ranks are 0) for tie-breaking.
                double best_possible_avg = (Cjk > 0) ? (next_sum_current / (double)Cjk) : 0.0;
                if (next_min_rank < threshold_rank || // Current min is already worse than threshold min
                   (next_min_rank == threshold_rank && best_possible_avg < threshold_secondary_rank)) // Or mins equal, and best avg is worse than threshold avg
                {
                    should_continue = 0; // Prune this branch
                }
            } else {
                // --- Pruning for 'avg' mode ---
                // Primary key: avg_rank (higher is better). Secondary: min_rank (higher is better).
                // Prune if the best possible average rank ('lower_bound_avg') can *never* beat the threshold.
                // Best possible average assumes all remaining subsets have rank 0 (best score).
                // Note: Rank 0 contributes 0 to the sum.
                double lower_bound_avg = (Cjk > 0) ? (next_sum_current / (double)Cjk) : 0.0;

                // Prune if the best possible average from this branch is worse than the threshold average,
                // considering the current minimum rank ('next_min_rank') for tie-breaking.
                if (lower_bound_avg < threshold_rank || // Best possible avg is already worse than threshold avg
                   (lower_bound_avg == threshold_rank && next_min_rank < threshold_secondary_rank)) // Or avgs equal, and current min is worse than threshold min
                {
                    should_continue = 0; // Prune this branch
                }
                // **Note:** This pruning for 'avg' is theoretically correct but may be less effective
                // than 'min' pruning in practice because the average rank converges slower than the minimum.
                // The lower bound used here is the most optimistic valid bound.
            }
        } // End pruning check

        // === Recurse if not pruned ===
        if (should_continue) {
            backtrack(S, size + 1, new_S, next_min_rank, next_sum_current, num + 1, // Use num+1 for next start
                      table, total_draws, max_number, j, k,
                      thread_best, thread_filled, l, m, Cjk);
        }
        // } // End redundant check for number existence
    } // End for loop over 'num'
}


// ----------------------------------------------------------------------
// Main Analysis Functions (Standard and Chain)
// ----------------------------------------------------------------------

/**
 * @brief Performs standard analysis: Finds top-L combos, then N non-overlapping from top-L.
 * Uses parallel backtracking search.
 */
static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data, // Pointer to start of relevant (newest) sorted draw data
    int use_count, // Number of draws to consider in this slice
    int j, int k, const char* m, int l, int n,
    int max_number, int* out_len)
{
    *out_len = 0; // Initialize output length

    // --- 1. Build Subset Hash Table ---
    // This table maps k-subset patterns to the relative index (0 to use_count-1) of the last draw they appeared in.
    SubsetTable* table = create_subset_table(HASH_SIZE);
    if (!table) {
        fprintf(stderr, "Error: Failed to create subset table.\n");
        return NULL;
    }
    // Process draws from the provided slice (assumed newest 'use_count' draws)
    for (int i = 0; i < use_count; i++) {
        // Pass draw pointer and its relative index 'i' (0 = oldest in window, use_count-1 = newest)
        process_draw(&sorted_draws_data[i * MAX_DRAW_SIZE], i, k, table);
    }

    // --- 2. Prepare for Parallel Backtracking ---
    int num_threads = omp_get_max_threads();
    // Allocate buffer to hold top-L results for each thread temporarily
    ComboStats* all_best_per_thread = (ComboStats*)malloc(num_threads * l * sizeof(ComboStats));
    if (!all_best_per_thread) {
        fprintf(stderr, "Error: Failed to allocate memory for thread results.\n");
        free_subset_table(table);
        return NULL;
    }
     // Initialize all slots to indicate they are empty/invalid (e.g., len=0)
    for(int i = 0; i < num_threads * l; ++i) {
        all_best_per_thread[i].len = 0;
        all_best_per_thread[i].avg_rank = -DBL_MAX; // Use sentinel values
        all_best_per_thread[i].min_rank = -DBL_MAX;
    }

    int error_occurred = 0; // Flag for critical errors (e.g., allocation failure) in threads
    // Precompute C(j, k) - total number of k-subsets per j-combination
    uint64 Cjk = (j >= k && k >= 0 && j <= MAX_NUMBERS && k <= MAX_NUMBERS) ? nCk_table[j][k] : 0;
    if (Cjk == 0 && k > 0 && j >=k) {
        fprintf(stderr, "Warning: C(%d, %d) is 0 or overflowed. Ranks may be calculated incorrectly.\n", j, k);
        // Proceed cautiously, avg rank calculation might yield NaN or Inf if Cjk is 0.
    }

    // --- 3. Parallel Backtracking Search ---
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int* S = (int*)malloc(j * sizeof(int)); // Thread-local buffer for current combination

        // Point thread_best to the correct segment in the shared buffer
        ComboStats* thread_best = &all_best_per_thread[thread_id * l];
        int thread_filled = 0; // Counter for valid entries in this thread's top-L list

        if (!S) {
            #pragma omp atomic write
            error_occurred = 1; // Signal allocation failure
        } else {
            // Note: thread_best is already initialized (len=0) outside parallel region

            // Distribute the starting number of combinations across threads dynamically
            #pragma omp for schedule(dynamic) nowait // nowait allows threads to proceed without implicit barrier
            for (int first_num = 1; first_num <= max_number - j + 1; first_num++) {
                 if (error_occurred) continue; // Stop work if another thread failed critically

                 S[0] = first_num; // Start combo with first_num
                 uint64 current_S = (first_num > 0 && first_num <= 64) ? (1ULL << (first_num - 1)) : 0;

                 // Initiate backtracking for combinations starting with first_num
                 // Initial min rank is worse than any possible rank. Initial sum is 0.
                 backtrack(S, 1, current_S, (double)(use_count + 1.0), 0.0, first_num + 1,
                           table, use_count, max_number, j, k,
                           thread_best, &thread_filled, l, m, Cjk);
            }
            free(S); // Free thread-local combination buffer
        }
    } // End of parallel region

    // Check for errors signaled by threads
    if (error_occurred) {
        free(all_best_per_thread);
        free_subset_table(table);
        return NULL;
    }

    // --- 4. Merge Results from all Threads into Final Top-L List ---
    ComboStats* final_best = (ComboStats*)malloc(l * sizeof(ComboStats));
     if (!final_best) {
         fprintf(stderr, "Error: Failed to allocate memory for final merge list.\n");
         free(all_best_per_thread);
         free_subset_table(table);
         return NULL;
     }
     // Initialize final_best list as empty
    for(int i = 0; i < l; ++i) {
         final_best[i].len = 0;
         final_best[i].avg_rank = -DBL_MAX;
         final_best[i].min_rank = -DBL_MAX;
    }
    int final_filled = 0; // Count of items in the final merged list

    // Iterate through all results potentially found by threads
    for (int i = 0; i < num_threads * l; ++i) {
        // Skip invalid entries (marked by len != j)
        if (all_best_per_thread[i].len != j) continue;

        // Use the helper function to merge into the final sorted list
        insert_into_best(final_best, l, &final_filled,
                         all_best_per_thread[i].avg_rank, all_best_per_thread[i].min_rank,
                         all_best_per_thread[i].combo, j,
                         all_best_per_thread[i].pattern, m);
    }
    free(all_best_per_thread); // Free the temporary thread results buffer

    // --- 5. Prepare Output Array (Format Top-L Results) ---
    // Allocate space for potentially L top results + N non-overlapping results
    AnalysisResultItem* results = (AnalysisResultItem*)calloc(l + n, sizeof(AnalysisResultItem));
    if (!results) {
        fprintf(stderr, "Error: Failed to allocate memory for output results array.\n");
        free(final_best);
        free_subset_table(table);
        return NULL;
    }

    int results_count = 0;
    // Format the top-L combos found (up to final_filled) into the output structure
    for (int i = 0; i < final_filled; i++) {
        if (results_count >= l + n) break; // Safety check against buffer overflow

        format_combo(final_best[i].combo, final_best[i].len, results[results_count].combination);
        // Re-use the subset table created earlier for formatting subset details
        format_subsets(final_best[i].combo, j, k, use_count, table, results[results_count].subsets);
        results[results_count].avg_rank = final_best[i].avg_rank;
        results[results_count].min_value = final_best[i].min_rank;
        // Initialize fields not relevant to standard analysis results
        results[results_count].is_chain_result = 0;
        results[results_count].draw_offset = 0;
        results[results_count].analysis_start_draw = 0;
        results[results_count].draws_until_common = 0;
        results_count++;
    }
    int top_l_count = results_count; // Record how many top-L items were added

    // --- 6. Find N Non-Overlapping Combos (if requested) ---
    if (n > 0 && final_filled > 0) {
        // Temp array to store indices (in final_best) of the chosen non-overlapping combos
        int* pick_indices = (int*)malloc(final_filled * sizeof(int));
        if (pick_indices) {
            int chosen_count = 0;
            // Always pick the best one (index 0 in final_best) first
            pick_indices[chosen_count++] = 0;

            // Iterate through the remaining top-L combos (from index 1)
            for (int i = 1; i < final_filled && chosen_count < n; i++) {
                uint64 pat_i = final_best[i].pattern; // Pattern of the candidate combo
                int overlap = 0;
                // Compare with combos already chosen for the non-overlapping set
                for (int c = 0; c < chosen_count; c++) {
                    int idxC = pick_indices[c]; // Index in final_best of a previously chosen combo
                    uint64 pat_c = final_best[idxC].pattern;
                    // Check for overlap: intersection contains k or more numbers
                    if (popcount64(pat_i & pat_c) >= k) {
                        overlap = 1;
                        break; // Overlaps with a chosen combo
                    }
                }
                // If no overlap found with any previously chosen combo, pick this one
                if (!overlap) {
                    pick_indices[chosen_count++] = i;
                }
            }

            // Format the chosen N non-overlapping results and append them to the output array
            int non_overlap_start_index = top_l_count; // Index where N results start
            for (int i = 0; i < chosen_count; i++) {
                 int idx = pick_indices[i]; // Index in final_best of the i-th non-overlapping combo

                 if (non_overlap_start_index + i >= l + n) break; // Safety check

                 // Format and copy data (similar to top-L loop)
                 format_combo(final_best[idx].combo, final_best[idx].len, results[non_overlap_start_index + i].combination);
                 format_subsets(final_best[idx].combo, j, k, use_count, table, results[non_overlap_start_index + i].subsets);
                 results[non_overlap_start_index + i].avg_rank = final_best[idx].avg_rank;
                 results[non_overlap_start_index + i].min_value = final_best[idx].min_rank;
                 results[non_overlap_start_index + i].is_chain_result = 0; // Mark as standard part
                 // Other fields remain 0
                 results_count++; // Increment total results count including these N items
            }
            free(pick_indices); // Free temp index array
        } else {
            fprintf(stderr, "Warning: Failed to allocate memory for non-overlapping indices.\n");
            // Proceed without adding N results if allocation failed
        }
    }

    // --- 7. Cleanup and Return ---
    *out_len = results_count; // Set the final number of items in the results array

    free_subset_table(table);
    free(final_best);

    if (results_count == 0) { // If no results were found at all
        free(results); // Free the allocated results array
        return NULL;
    }

    return results; // Return the populated array of results
}

/**
 * @brief Performs chain analysis: Finds top-1 combo iteratively over shrinking draw windows.
 */
static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data, // Contains ALL sorted draws, oldest to newest
    int draws_count,              // Total number of draws available
    int initial_offset,           // Initial number of draws to exclude from the end
    int j, int k, const char* m,
    int max_number, int* out_len)
{
    *out_len = 0;
    // Allocate space for results: Max iterations is roughly initial_offset + 1. Add buffer.
    AnalysisResultItem* chain_results = (AnalysisResultItem*)calloc(initial_offset + 2, sizeof(AnalysisResultItem));
    if (!chain_results) {
        fprintf(stderr, "Error: Failed to allocate memory for chain results.\n");
        return NULL;
    }

    // Precompute bit patterns for all draws for faster overlap checking later
    uint64* draw_patterns = (uint64*)malloc(draws_count * sizeof(uint64));
    if (!draw_patterns) {
        fprintf(stderr, "Error: Failed to allocate memory for draw patterns.\n");
        free(chain_results);
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        draw_patterns[i] = numbers_to_pattern(&sorted_draws_data[i * MAX_DRAW_SIZE], MAX_DRAW_SIZE);
    }

    int chain_index = 0; // Index for storing results in chain_results array
    int current_offset = initial_offset; // Current number of draws excluded from newest end
    uint64 Cjk = (j >= k && k >= 0 && j <= MAX_NUMBERS && k <= MAX_NUMBERS) ? nCk_table[j][k] : 0;

    // Loop until offset becomes invalid or results buffer fills
    while (current_offset >= 0 && current_offset < draws_count) {
        int use_count = draws_count - current_offset; // Number of newest draws to use for this step
        // Need at least k draws to form k-subsets. Need j numbers < max_number.
        if (use_count < k || j > max_number) break;

        // --- 1. Build Subset Table for the Current Window ---
        SubsetTable* table = create_subset_table(HASH_SIZE);
         if (!table) { fprintf(stderr, "Error: Failed to create subset table in chain loop.\n"); break; }
        // Determine the absolute index of the first draw in this window
        int first_draw_idx_in_window = draws_count - use_count;
        // Process draws within the window, using relative index 0 to use_count-1
        for (int i = 0; i < use_count; i++) {
            process_draw(&sorted_draws_data[(first_draw_idx_in_window + i) * MAX_DRAW_SIZE], i, k, table);
        }

        // --- 2. Find Top-1 Combo for this Window (using Backtrack with l=1) ---
        ComboStats best_chain_combo; // Structure to hold the single best combo
        best_chain_combo.len = 0; // Mark as empty/invalid initially
        best_chain_combo.avg_rank = -DBL_MAX;
        best_chain_combo.min_rank = -DBL_MAX;
        int filled_chain = 0; // Counter, will be 0 or 1

        int* S_chain = (int*)malloc(j * sizeof(int)); // Temp buffer for backtracking
        if (!S_chain) {
            fprintf(stderr, "Error: Failed to allocate memory for backtrack buffer in chain loop.\n");
            free_subset_table(table); break;
        }

        // Iterate through all possible starting numbers for the combination
        for (int first_num = 1; first_num <= max_number - j + 1; first_num++) {
             S_chain[0] = first_num;
             uint64 current_S = (first_num > 0 && first_num <= 64) ? (1ULL << (first_num - 1)) : 0;
             // Start backtracking with l=1 to find only the single best combo
             backtrack(S_chain, 1, current_S, (double)(use_count + 1.0), 0.0, first_num + 1,
                       table, use_count, max_number, j, k,
                       &best_chain_combo, &filled_chain, 1, m, Cjk); // l=1
        }
        free(S_chain); // Free temp buffer

        // Check if a valid top-1 combo was found for this window
        if (filled_chain == 0 || best_chain_combo.len != j) {
            //fprintf(stderr, "Debug: No valid combo found for offset %d, use_count %d. Stopping chain.\n", current_offset, use_count);
            free_subset_table(table);
            break; // Stop the chain if no combo is found
        }

        // --- 3. Store Found Combo Results ---
        // Safety check against exceeding allocated results buffer
        if (chain_index >= initial_offset + 2) {
            fprintf(stderr, "Warning: Exceeded allocated chain results buffer size.\n");
            free_subset_table(table); break;
        }
        AnalysisResultItem* out_item = &chain_results[chain_index];
        format_combo(best_chain_combo.combo, best_chain_combo.len, out_item->combination);
        format_subsets(best_chain_combo.combo, j, k, use_count, table, out_item->subsets); // Reuse table for format
        out_item->avg_rank = best_chain_combo.avg_rank;
        out_item->min_value = best_chain_combo.min_rank;
        out_item->is_chain_result = 1; // Mark as chain result
        out_item->draw_offset = chain_index + 1; // "Analysis #" (1-based index of this chain step)
        out_item->analysis_start_draw = first_draw_idx_in_window + 1; // "For Draw" (1-based index of first draw in window)

        // --- 4. Find Duration (draws until common k-subset with a future draw) ---
        uint64 combo_pat = best_chain_combo.pattern;
        int steps_forward;
        // Check draws *after* the current window, up to 'current_offset' steps forward
        int last_draw_idx_in_window = draws_count - current_offset -1; // This seems off. It's draws_count - 1 - current_offset? NO.
        // Last draw used has absolute index: first_draw_idx_in_window + use_count - 1 = (draws_count - use_count) + use_count - 1 = draws_count - 1
        // The first draw *outside* the window (going forward in time) is index: draws_count - current_offset
        for (steps_forward = 1; steps_forward <= current_offset; steps_forward++) {
            // Calculate the absolute index of the draw 'steps_forward' ahead
            int future_draw_abs_idx = (draws_count - current_offset) + steps_forward - 1;
            if (future_draw_abs_idx >= draws_count) break; // Ensure index is valid

            uint64 fpat = draw_patterns[future_draw_abs_idx]; // Get precomputed pattern
            if (popcount64(combo_pat & fpat) >= k) { // Check for k-subset overlap
                break; // Found overlap 'steps_forward' steps ahead
            }
        }
        // If loop finished without break, overlap wasn't found within 'current_offset' steps.
        // steps_forward will be current_offset + 1 in this case.
        // The 'duration' is the number of draws *before* the overlap occurred (0-based).
        out_item->draws_until_common = (steps_forward > 0) ? (steps_forward - 1) : 0;

        // --- 5. Update Offset for Next Iteration ---
        // Decrease offset by the number of steps until overlap was found (or current_offset+1 if not found)
        current_offset -= steps_forward;
        chain_index++; // Move to next slot in results array

        free_subset_table(table); // Free table for this iteration

        // Check if next iteration is possible
        if (current_offset < 0) break;

    } // End while loop for chain iterations

    // --- Cleanup and Set Output Length ---
    free(draw_patterns);
    *out_len = chain_index; // Number of chain results actually generated

    if (chain_index == 0) { // If no results generated at all
        free(chain_results);
        return NULL;
    }
    // Return the (potentially partially filled) results array
    return chain_results;
}


// ----------------------------------------------------------------------
// Main Entry Point (Called from Python via CTypes)
// ----------------------------------------------------------------------
/**
 * @brief Main entry point for the C analysis engine. Dispatches to standard or chain analysis.
 * @param game_type String indicating game type (e.g., "6_49", used for max_number).
 * @param draws Jagged array of draw numbers (draws_count x MAX_DRAW_SIZE).
 * @param draws_count Total number of draws provided.
 * @param j Size of combinations to find.
 * @param k Size of subsets to analyze.
 * @param m Sorting mode ("avg" or "min").
 * @param l Top-L count for standard analysis (-1 for chain analysis).
 * @param n N non-overlapping count for standard analysis (ignored for chain).
 * @param last_offset Offset from newest draw (standard) or initial offset (chain).
 * @param out_len Pointer to integer where the number of results returned will be stored.
 * @return Pointer to an array of AnalysisResultItem structures, or NULL on error/no results.
 *         The caller is responsible for freeing this memory using free_analysis_results.
 */
AnalysisResultItem* run_analysis_c(
    const char* game_type, int** draws, int draws_count,
    int j, int k, const char* m, int l, int n,
    int last_offset, int* out_len)
{
    *out_len = 0; // Initialize output length

    // --- Basic Input Validation ---
    if (!game_type || !draws || !m || !out_len || draws_count <= 0 || j <= 0 || k <= 0 || j < k || j > MAX_ALLOWED_J || k > MAX_DRAW_SIZE) {
        fprintf(stderr, "Error: Invalid arguments passed to run_analysis_c.\n");
        fprintf(stderr, "  game_type=%p, draws=%p, draws_count=%d, j=%d, k=%d, m=%p, l=%d, n=%d, last_offset=%d, out_len=%p\n",
                (void*)game_type, (void*)draws, draws_count, j, k, (void*)m, l, n, last_offset, (void*)out_len);
        if (j < k) fprintf(stderr, "  Reason: j (%d) < k (%d)\n", j, k);
        if (j > MAX_ALLOWED_J) fprintf(stderr, "  Reason: j (%d) > MAX_ALLOWED_J (%d)\n", j, MAX_ALLOWED_J);
        if (k > MAX_DRAW_SIZE) fprintf(stderr, "  Reason: k (%d) > MAX_DRAW_SIZE (%d)\n", k, MAX_DRAW_SIZE);
        return NULL;
    }
    if (strcmp(m, "avg") != 0 && strcmp(m, "min") != 0) {
        fprintf(stderr, "Error: Invalid sort mode '%s'. Must be 'avg' or 'min'.\n", m);
        return NULL;
    }
     if (l != -1 && (l <= 0 || n < 0)) { // Standard analysis requires l > 0 and n >= 0
        fprintf(stderr, "Error: Invalid l (%d) or n (%d) for standard analysis (l > 0, n >= 0 required).\n", l, n);
        return NULL;
     }
     if (l == -1) n = 0; // n parameter is ignored in chain analysis

    // --- Initialization ---
    init_tables(); // Ensure nCk table etc. are ready (safe to call multiple times)

    // Determine max number based on game type (simple string check)
    int max_number = (strstr(game_type, "49")) ? 49 : 42; // Default to 42 if "49" not found
    if (max_number <= 0 || max_number > MAX_NUMBERS) {
        fprintf(stderr, "Error: Determined max_number %d is invalid or exceeds MAX_NUMBERS %d.\n", max_number, MAX_NUMBERS);
        return NULL;
    }
    if (j > max_number) {
         fprintf(stderr, "Error: Combination size j (%d) cannot exceed max_number (%d).\n", j, max_number);
         return NULL;
    }


    // --- Validate Offset ---
    if (last_offset < 0) last_offset = 0;
    // Ensure offset doesn't exclude all draws (need at least 1 for standard, or k for chain implicitly)
    if (last_offset >= draws_count) {
         if (draws_count > 0) {
            last_offset = draws_count - 1; // Adjust to keep at least one draw
            fprintf(stderr, "Warning: Offset %d reduced to %d to include at least one draw.\n", last_offset+1, last_offset);
         } else {
            fprintf(stderr, "Error: No draws provided (draws_count=0).\n");
            return NULL; // No draws to analyze
         }
    }

    // --- Prepare Draw Data ---
    // Create a contiguous, sorted copy of the draw data.
    // Input 'draws' is oldest (index 0) to newest (draws_count - 1). We preserve this order.
    int* sorted_draws_data = (int*)malloc(draws_count * MAX_DRAW_SIZE * sizeof(int));
    if (!sorted_draws_data) {
        fprintf(stderr, "Error: Failed to allocate memory for sorted draw data (%d draws).\n", draws_count);
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        int temp[MAX_DRAW_SIZE];
        // Check if draw pointer is valid before copying
        if (!draws[i]) {
             fprintf(stderr, "Error: Invalid draw data pointer at index %d.\n", i);
             free(sorted_draws_data); return NULL;
        }
        memcpy(temp, draws[i], MAX_DRAW_SIZE * sizeof(int));

        // Simple insertion sort (efficient for small, fixed size like 6)
        for(int ii = 1; ii < MAX_DRAW_SIZE; ++ii) {
            int key = temp[ii];
            int jj = ii - 1;
            // Sort ascending
            while(jj >= 0 && temp[jj] > key) {
                temp[jj + 1] = temp[jj];
                jj = jj - 1;
            }
            temp[jj + 1] = key;
        }
        // Copy sorted draw into the contiguous array
        memcpy(&sorted_draws_data[i * MAX_DRAW_SIZE], temp, MAX_DRAW_SIZE * sizeof(int));
    }

    // --- Dispatch to Appropriate Analysis Function ---
    AnalysisResultItem* ret = NULL;
    if (l != -1) { // Standard Analysis (Top-L and N non-overlapping)
        int use_count = draws_count - last_offset; // Number of newest draws to use
        if (use_count <= 0) {
             fprintf(stderr, "Error: Offset %d results in non-positive number of draws (%d) for standard analysis.\n", last_offset, use_count);
             free(sorted_draws_data);
             return NULL;
        }
        // Pointer to the start of the relevant draw data slice (newest 'use_count' draws)
        const int* relevant_draws_ptr = &sorted_draws_data[(draws_count - use_count) * MAX_DRAW_SIZE];

        ret = run_standard_analysis(relevant_draws_ptr, use_count, j, k, m, l, n, max_number, out_len);

    } else { // Chain Analysis (l == -1)
        // Chain analysis uses all draws but starts with an initial offset
        ret = run_chain_analysis(sorted_draws_data, draws_count, last_offset, j, k, m, max_number, out_len);
    }

    // --- Cleanup ---
    free(sorted_draws_data); // Free the copied draw data
    return ret; // Return results (or NULL if error/no results)
}

// ----------------------------------------------------------------------
// Memory Freeing Function (Called from Python)
// ----------------------------------------------------------------------
/**
 * @brief Frees the memory allocated for the array of AnalysisResultItem structures.
 * @param results Pointer to the array returned by run_analysis_c.
 */
void free_analysis_results(AnalysisResultItem* results) {
    if (results) {
        free(results);
    }
}

// ----------------------------------------------------------------------
// Formatting Functions (Implementations)
// ----------------------------------------------------------------------

/**
 * @brief Formats a combination (array of numbers) into a comma-separated string.
 * Ensures null termination and prevents buffer overflows.
 */
static void format_combo(const int* combo, int len, char* out) {
    if (!out) return;
    int pos = 0;
    int remaining = MAX_COMBO_STR; // Use buffer size constant

    for (int i = 0; i < len; i++) {
        int written;
        const char* format_str = (i == 0) ? "%d" : ", %d"; // Format string
        int space_needed = (i == 0) ? 1 : 3; // Approx space for format + number

        if (remaining < space_needed) { // Basic check if space is too tight
            pos = MAX_COMBO_STR - 1; break;
        }

        written = snprintf(out + pos, remaining, format_str, combo[i]);

        // Check for snprintf errors or truncation
        if (written < 0 || written >= remaining) {
            pos = MAX_COMBO_STR - 1; // Ensure space for null terminator
            break;
        }
        pos += written;
        remaining -= written;
    }
    // Ensure null termination even if loop finishes early or buffer is full
    out[pos < MAX_COMBO_STR ? pos : MAX_COMBO_STR - 1] = '\0';
}

/**
 * @brief Formats the list of k-subsets of a combination and their ranks into a string.
 * Subsets are sorted by rank (score) descending. Matches original format.
 * Handles potential errors and limits processing for very large numbers of subsets.
 */
static void format_subsets(const int* combo, int j, int k, int total_draws,
                          const SubsetTable* table, char* out)
{
    if (!out) return; // Check output buffer pointer
    out[0] = '\0'; // Initialize output buffer

    // --- Basic Input Validation ---
    if (k <= 0 || k > j || j > MAX_ALLOWED_J || k > MAX_NUMBERS || !combo || !table) {
         snprintf(out, MAX_SUBSETS_STR, "[]"); // Empty list for invalid input
         return;
    }

    // --- Calculate Number of Subsets and Check Limits ---
    uint64 exact_subset_count_64 = nCk_table[j][k];
    if (exact_subset_count_64 == 0) {
        snprintf(out, MAX_SUBSETS_STR, "[]"); // No subsets if C(j,k) is 0
        return;
    }
    // Limit processing to avoid excessive memory/time for huge C(j, k)
    const uint64 MAX_PROCESS_SUBSETS = 1000000;
    if (exact_subset_count_64 > MAX_PROCESS_SUBSETS) {
        snprintf(out, MAX_SUBSETS_STR, "[Too many subsets C(%d,%d) > %llu to format]", j, k, MAX_PROCESS_SUBSETS);
        return;
    }
    int exact_subset_count = (int)exact_subset_count_64;

    // --- Allocate Temporary Storage for Subsets ---
    typedef struct { int numbers[MAX_NUMBERS]; int rank; } SubsetInfo; // Use MAX_NUMBERS defensively? No, k.
    if (k > MAX_NUMBERS) { /* Should have been caught earlier */ snprintf(out, MAX_SUBSETS_STR, "[]"); return; }

    // Allocate memory to store subset numbers and their ranks
    SubsetInfo* subsets = (SubsetInfo*)malloc(exact_subset_count * sizeof(SubsetInfo));
    if (!subsets) {
        snprintf(out, MAX_SUBSETS_STR, "[Memory allocation error for subsets]");
        return;
    }
    int subset_count = 0; // Counter for subsets processed

    // --- Generate and Rank All k-Subsets ---
    int idx[MAX_ALLOWED_J]; // Buffer for indices into 'combo' array, size k is needed
    for (int i = 0; i < k; i++) idx[i] = i; // Initialize first combination indices

    while (subset_count < exact_subset_count) {
        // Form the current k-subset numbers and its bit pattern
        uint64 pat = 0ULL;
        int current_subset_nums[k]; // Temp storage for numbers in this subset
        for (int i = 0; i < k; i++) {
            int num = combo[idx[i]];
            current_subset_nums[i] = num;
            if (num > 0 && num <= 64) { // Check bounds for bitmask
                pat |= (1ULL << (num - 1));
            }
        }
        // Copy numbers to the storage array
        memcpy(subsets[subset_count].numbers, current_subset_nums, k * sizeof(int));

        // Lookup rank (score) from hash table
        int last_seen = lookup_subset(table, pat);
        int rank = (last_seen >= 0) ? (total_draws - last_seen - 1) : total_draws;
        subsets[subset_count].rank = rank;
        subset_count++;

        // Generate next combination of indices (standard algorithm C(j, k))
        int p = k - 1; // Start from the rightmost index
        while (p >= 0 && idx[p] == j - k + p) { // Find rightmost index that can be incremented
            p--;
        }
        if (p < 0) break; // Last combination generated
        idx[p]++; // Increment the index
        for (int x = p + 1; x < k; x++) { // Reset subsequent indices
            idx[x] = idx[x - 1] + 1;
        }
    } // End while loop generating subsets

    // --- Sort Subsets by Rank (Score) Descending ---
    // Define comparison function for qsort
    int compare_subset_rank_desc(const void* a, const void* b) {
         int rank_a = ((SubsetInfo*)a)->rank;
         int rank_b = ((SubsetInfo*)b)->rank;
         // Higher rank (score) should come first
         if (rank_a > rank_b) return -1;
         if (rank_a < rank_b) return 1;
         // Optional: Add secondary sort by numbers if ranks are equal? Original didn't specify.
         // For stability and matching original potential behavior, compare number arrays if ranks equal.
         const int* nums_a = ((SubsetInfo*)a)->numbers;
         const int* nums_b = ((SubsetInfo*)b)->numbers;
         for(int i=0; i<k; ++i) {
             if (nums_a[i] < nums_b[i]) return -1; // Lower number comes first for tie-break
             if (nums_a[i] > nums_b[i]) return 1;
         }
         return 0; // Ranks and numbers are identical
    }
    // Sort the array using qsort
    qsort(subsets, subset_count, sizeof(SubsetInfo), compare_subset_rank_desc);

    // --- Format Sorted Subsets into Output String ---
    int pos = 0;
    int remaining_space = MAX_SUBSETS_STR - 1; // Leave space for null terminator

    if (remaining_space <= 0) { free(subsets); out[0] = '\0'; return; } // Check initial space
    out[pos++] = '['; remaining_space--;

    for (int i = 0; i < subset_count && remaining_space > 0; i++) {
        if (i > 0) { // Add separator between subsets
            if (remaining_space < 2) break;
            out[pos++] = ','; out[pos++] = ' ';
            remaining_space -= 2;
        }

        // Format single subset entry: "((num1, num2, ...), rank)"
        if (remaining_space < 3) break; // Need space for "((" minimum
        out[pos++] = '('; out[pos++] = '('; remaining_space -= 2;

        // Format numbers within the subset parentheses
        for (int n = 0; n < k && remaining_space > 0; n++) {
            int written;
            const char* num_format = (n == 0) ? "%d" : ", %d";
            int space_needed = (n == 0) ? 1 : 3; // Approx space needed
            if (remaining_space < space_needed) { remaining_space = 0; break; }

            written = snprintf(out + pos, remaining_space, num_format, subsets[i].numbers[n]);
            if (written < 0 || written >= remaining_space) { remaining_space = 0; break; }
            pos += written; remaining_space -= written;
        }
        if (remaining_space == 0) break;

        // Format the closing parenthesis and rank: "), rank)"
        if (remaining_space < 4) break; // Need space for "), R)"
        int written = snprintf(out + pos, remaining_space, "), %d)", subsets[i].rank);
        if (written < 0 || written >= remaining_space) { remaining_space = 0; break; }
        pos += written; remaining_space -= written;
        if (remaining_space == 0) break;
    }

    // Add closing bracket for the list
    if (remaining_space > 0) {
        out[pos++] = ']';
    } else if (pos < MAX_SUBSETS_STR && pos > 0) { // If buffer filled exactly or overflowed minimally
        out[MAX_SUBSETS_STR - 2] = ']'; // Try to force ']' at the end
        pos = MAX_SUBSETS_STR - 1;      // Position for null terminator
    } else if (pos == 0) { // If buffer was too small even for '['
        out[0] = '['; out[1] = ']'; out[2] = '\0'; pos=2; // Output just "[]"
    } else { // Ensure pos is valid for null term if completely overflowed
        pos = MAX_SUBSETS_STR - 1;
    }
    out[pos] = '\0'; // Null terminate the string

    free(subsets); // Free the temporary subset array
}
