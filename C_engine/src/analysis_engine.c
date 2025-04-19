// Filename: C_engine/src/analysis_engine.c
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h> // For INT_MAX if needed, though using double thresholds now
#include <float.h> // For DBL_MAX, -INFINITY

#include "analysis_engine.h"

#define MAX_COMBO_STR 256       // Increased slightly for safety
#define MAX_SUBSETS_STR 65536   // Use K instead of fixed size
#define MAX_ALLOWED_J 200
#define MAX_ALLOWED_OUT_LEN 1000000 // Max results array size
#define MAX_NUMBERS 50          // Max number in any game (e.g., 49 for 6/49)
#define HASH_SIZE (1 << 26)     // 67M entries, adjust based on memory/performance trade-off

typedef unsigned long long uint64;
typedef unsigned int uint32;

// --- Global Lookups ---
static uint64 nCk_table[MAX_NUMBERS + 1][MAX_NUMBERS + 1]; // Use MAX_NUMBERS+1 size
static int bit_count_table[256]; // For popcount optimization
static int initialized = 0;

// --- Data Structures ---
typedef struct {
    uint64* keys;     // Subset bit patterns (hash table keys)
    int* values;      // Last draw index seen (hash table values)
    // Removed size/capacity, using fixed-size hash table with collision handling
} SubsetTable; // Simplified to just pointers for fixed-size hash table

typedef struct {
    uint64 pattern;         // Bit pattern of the combo's numbers
    double avg_rank;
    double min_rank;
    int combo[MAX_NUMBERS]; // Store the combination numbers
    int len;                // Should always be 'j' for valid entries
} ComboStats;

// --- Forward Declarations ---
static void init_tables();
static inline int popcount64(uint64 x);
static SubsetTable* create_subset_table(); // Simplified
static void free_subset_table(SubsetTable* table);
static void clear_subset_table(SubsetTable* table); // To reuse hash table
static inline uint32 hash_subset(uint64 pattern);
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value);
static inline int lookup_subset(const SubsetTable* table, uint64 pattern);
static inline uint64 numbers_to_pattern(const int* numbers, int count);
static void process_draw(const int* draw_numbers, int draw_idx, int k, SubsetTable* table); // Takes 6 numbers
static void format_combo(const int* combo, int len, char* out);
static void format_subsets(const int* combo, int j, int k, int total_draws,
                           const SubsetTable* table, char* out);

static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data_all, // Pointer to all sorted draw data
    int use_count, // Number of draws to actually use (from the start)
    int j,
    int k,
    const char* m,
    int l,
    int n,
    int max_number,
    int* out_len
);

static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data_all, // Pointer to all sorted draw data
    int draws_count, // Total number of draws available
    int initial_offset, // Starting offset for the first analysis
    int j,
    int k,
    const char* m,
    int max_number,
    int* out_len
);

// The core recursive function for finding top-L combinations
static void backtrack(
    // Current state
    int* S,                 // Array holding the current partial combination
    int size,               // Current number of elements in S
    uint64 current_S_pattern,// Bit pattern for the current partial combination S
    double current_min_rank,// Minimum rank encountered among all k-subsets of S so far
    double sum_ranks_so_far,// Sum of ranks for all k-subsets generated from S so far
    int start_num,          // Next number to consider adding to S
    // Context & Parameters
    const SubsetTable* table, // Hash table mapping k-subset patterns to last seen draw index
    int total_draws,        // Total number of draws being considered (use_count)
    int max_number,         // Maximum possible number in the lottery (e.g., 42, 49)
    int j,                  // Target size of the combination
    int k,                  // Size of subsets to check
    uint64 Cjk,             // Precalculated C(j, k) - total subsets in a full j-combo
    const char* m,          // Sorting mode ("avg" or "min")
    int l,                  // Number of top combinations to keep
    // Output / Thread-local storage
    ComboStats* thread_best,// Array to store the top L combinations found by this thread
    int* thread_filled      // Pointer to the count of valid entries in thread_best
);

// Comparison functions for qsort (if needed, but manual sort used here)
// static int compare_avg_rank(const void* a, const void* b);
// static int compare_min_rank(const void* a, const void* b);


// --- Initialization ---
static void init_tables() {
    if (initialized) return;

    // nCk Table (Pascal's Triangle)
    memset(nCk_table, 0, sizeof(nCk_table));
    for (int n = 0; n <= MAX_NUMBERS; n++) {
        nCk_table[n][0] = 1;
        for (int kk = 1; kk <= n; kk++) {
            // Prevent overflow if intermediate values exceed uint64 capacity
            uint64 term1 = (kk > 0) ? nCk_table[n - 1][kk - 1] : 0;
            uint64 term2 = nCk_table[n - 1][kk];
            if (UINT64_MAX - term1 < term2) { // Check for potential overflow
                 nCk_table[n][kk] = UINT64_MAX; // Mark as overflow / unusable large value
            } else {
                 nCk_table[n][kk] = term1 + term2;
            }
        }
    }

    // Bit Count Table (for popcount)
    for (int i = 0; i < 256; i++) {
        int c = 0;
        for (int b = 0; b < 8; b++) {
            if (i & (1 << b)) c++;
        }
        bit_count_table[i] = c;
    }

    initialized = 1;
}

// --- Utilities ---
static inline int popcount64(uint64 x) {
    // Use builtin if available (faster)
    #ifdef __GNUC__
        return __builtin_popcountll(x);
    #else
        // Fallback using lookup table
        int count = 0;
        count += bit_count_table[(x >> 0) & 0xFF];
        count += bit_count_table[(x >> 8) & 0xFF];
        count += bit_count_table[(x >> 16) & 0xFF];
        count += bit_count_table[(x >> 24) & 0xFF];
        count += bit_count_table[(x >> 32) & 0xFF];
        count += bit_count_table[(x >> 40) & 0xFF];
        count += bit_count_table[(x >> 48) & 0xFF];
        count += bit_count_table[(x >> 56) & 0xFF];
        return count;
    #endif
}

static SubsetTable* create_subset_table() {
    SubsetTable* t = (SubsetTable*)malloc(sizeof(SubsetTable));
    if (!t) return NULL;
    // Allocate fixed size hash table, use calloc for zero-initialization
    t->keys = (uint64*)calloc(HASH_SIZE, sizeof(uint64));
    t->values = (int*)malloc(HASH_SIZE * sizeof(int));
    if (!t->keys || !t->values) {
        free(t->keys);
        free(t->values);
        free(t);
        return NULL;
    }
    // Initialize values to -1 (indicating empty slot)
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

// Optional: Function to clear table for reuse (e.g., in chain analysis)
static void clear_subset_table(SubsetTable* table) {
    if (!table) return;
    // Reset keys to 0 and values to -1
    memset(table->keys, 0, HASH_SIZE * sizeof(uint64));
    for (int i = 0; i < HASH_SIZE; i++) {
        table->values[i] = -1;
    }
}


// Simple multiplicative hash function (adjust constants if needed)
static inline uint32 hash_subset(uint64 pattern) {
    pattern = (pattern ^ (pattern >> 32)) * 0x4cf5ad432745937fULL; // 64-bit mixing constant
    pattern = (pattern ^ (pattern >> 29)) * 0x515dd46f064b5ef5ULL;
    return (uint32)(pattern ^ (pattern >> 32)) & (HASH_SIZE - 1); // Final mix and mask
}

// Insert using linear probing for collision resolution
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value) {
    uint32 idx = hash_subset(pattern);
    uint32 start_idx = idx;
    while (1) {
        // Found empty slot or the key itself?
        if (table->values[idx] == -1 || table->keys[idx] == pattern) {
            table->keys[idx] = pattern;
            table->values[idx] = value; // Update with latest value (draw index)
            return;
        }
        // Move to next slot (wrap around)
        idx = (idx + 1) & (HASH_SIZE - 1);
        // Check if we've wrapped around completely (table full and key not found)
        if (idx == start_idx) {
             // This should ideally not happen if HASH_SIZE is large enough
             // Handle error: e.g., fprintf(stderr, "Hash table full!\n"); return;
             return; // Or resize, but we use fixed size here.
        }
    }
}

// Lookup using linear probing
static inline int lookup_subset(const SubsetTable* table, uint64 pattern) {
    uint32 idx = hash_subset(pattern);
    uint32 start_idx = idx;
    while (1) {
        if (table->values[idx] == -1) return -1; // Empty slot found, key not present
        if (table->keys[idx] == pattern) return table->values[idx]; // Key found
        // Move to next slot
        idx = (idx + 1) & (HASH_SIZE - 1);
        // Check if we've wrapped around
        if (idx == start_idx) return -1; // Key not found after full circle
    }
}

// Convert array of numbers to bit pattern (numbers assumed 1-based)
static inline uint64 numbers_to_pattern(const int* numbers, int count) {
    uint64 p = 0ULL;
    for (int i = 0; i < count; i++) {
        // Ensure number is within valid range (1 to 64) before shifting
        if (numbers[i] >= 1 && numbers[i] <= 64) {
             p |= (1ULL << (numbers[i] - 1));
        }
    }
    return p;
}


// Process a single draw: find all k-subsets and add/update in hash table
static void process_draw(const int* draw_numbers, int draw_idx, int k, SubsetTable* table) {
    if (k < 1 || k > 6) return; // k must be between 1 and 6 (draw size)

    int current_subset_indices[6]; // Indices into draw_numbers array (0 to 5)
    for (int i = 0; i < k; i++) current_subset_indices[i] = i;

    while (1) {
        // Create the k-subset pattern using the current indices
        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) {
            int num_index = current_subset_indices[i];
             // Ensure number is within valid range (1 to 64) before shifting
            if (draw_numbers[num_index] >= 1 && draw_numbers[num_index] <= 64) {
                pat |= (1ULL << (draw_numbers[num_index] - 1));
            }
        }
        // Insert/update the pattern in the hash table with the current draw index
        insert_subset(table, pat, draw_idx);

        // Generate the next combination of indices (standard algorithm for C(6, k))
        int p = k - 1; // Start from the rightmost index
        // Find the rightmost index that can be incremented
        while (p >= 0 && current_subset_indices[p] == 6 - k + p) {
            p--;
        }

        if (p < 0) break; // No more combinations possible

        // Increment the found index
        current_subset_indices[p]++;

        // Reset subsequent indices
        for (int x = p + 1; x < k; x++) {
            current_subset_indices[x] = current_subset_indices[x - 1] + 1;
        }
    }
}

// --- Backtracking Core Logic ---
static void backtrack(
    int* S,
    int size,
    uint64 current_S_pattern,
    double current_min_rank,
    double sum_ranks_so_far,
    int start_num,
    const SubsetTable* table,
    int total_draws, // This is use_count
    int max_number,
    int j,
    int k,
    uint64 Cjk,     // Precalculated C(j, k)
    const char* m,
    int l,
    ComboStats* thread_best,
    int* thread_filled
) {
    // --- Base Case: Combination Complete ---
    if (size == j) {
        // Calculate final avg and min ranks
        double avg_rank = (Cjk > 0) ? (sum_ranks_so_far / (double)Cjk) : 0.0; // Avoid division by zero if Cjk=0
        double min_rank = current_min_rank;

        // Check if this combo should be inserted into the top-L list
        int should_insert = 0;
        if (*thread_filled < l) {
            should_insert = 1; // List not full yet
        } else {
            // Compare with the worst element in the current top-L list
            if (strcmp(m, "avg") == 0) {
                should_insert = (avg_rank > thread_best[l - 1].avg_rank) ||
                                (avg_rank == thread_best[l - 1].avg_rank && min_rank > thread_best[l - 1].min_rank);
            } else { // m == "min"
                should_insert = (min_rank > thread_best[l - 1].min_rank) ||
                                (min_rank == thread_best[l - 1].min_rank && avg_rank > thread_best[l - 1].avg_rank);
            }
        }

        if (should_insert) {
            // Find insertion position and shift elements if necessary
            int insert_pos = l - 1; // Start checking from the end
            if (*thread_filled < l) {
                insert_pos = *thread_filled; // Append if list not full
                (*thread_filled)++;
            }

            // Shift elements down to make space for the new combo
            while (insert_pos > 0) {
                int compare_to_prev = 0; // Should we compare to thread_best[insert_pos - 1]?
                if (strcmp(m, "avg") == 0) {
                    compare_to_prev = (avg_rank > thread_best[insert_pos - 1].avg_rank) ||
                                      (avg_rank == thread_best[insert_pos - 1].avg_rank && min_rank > thread_best[insert_pos - 1].min_rank);
                } else { // m == "min"
                    compare_to_prev = (min_rank > thread_best[insert_pos - 1].min_rank) ||
                                      (min_rank == thread_best[insert_pos - 1].min_rank && avg_rank > thread_best[insert_pos - 1].avg_rank);
                }

                if (compare_to_prev) { // Current combo is better than the one at insert_pos - 1
                    memcpy(&thread_best[insert_pos], &thread_best[insert_pos - 1], sizeof(ComboStats));
                    insert_pos--; // Move comparison position up
                } else {
                    break; // Found the correct insertion spot
                }
            }

            // Insert the new combination
            if (insert_pos >= 0 && insert_pos < l) { // Ensure valid index
               memcpy(thread_best[insert_pos].combo, S, j * sizeof(int));
               thread_best[insert_pos].len = j;
               thread_best[insert_pos].avg_rank = avg_rank;
               thread_best[insert_pos].min_rank = min_rank;
               thread_best[insert_pos].pattern = current_S_pattern;
            }
        }
        return; // End recursion for this branch
    }

    // --- Pruning Threshold Check ---
    // Get the thresholds from the *worst* combo currently in the top-L list
    double threshold_primary = -INFINITY;
    double threshold_secondary = -INFINITY;
    int list_is_full = (*thread_filled == l);

    if (list_is_full) {
        if (strcmp(m, "avg") == 0) {
            threshold_primary = thread_best[l - 1].avg_rank;
            threshold_secondary = thread_best[l - 1].min_rank;
        } else { // m == "min"
            threshold_primary = thread_best[l - 1].min_rank;
            threshold_secondary = thread_best[l - 1].avg_rank;
        }
    }

    // --- Recursive Step: Iterate through possible next numbers ---
    // Optimization: Calculate remaining slots needed
    int remaining_slots = j - size;
    for (int num = start_num; num <= max_number - remaining_slots + 1; num++) { // +1 because num is inclusive

         // Add the number 'num' to the current partial combo S
         S[size] = num;
         uint64 next_S_pattern = current_S_pattern | (1ULL << (num - 1));

         // Calculate rank contributions ONLY for the NEW subsets formed by adding 'num'
         double min_rank_of_new_subsets = (double)(total_draws + 1.0); // Initialize worst possible
         double sum_ranks_of_new_subsets = 0.0;
         uint64 num_new_subsets_formed = 0; // How many k-subsets include 'num'

         // We only need to check subsets if we have enough elements (size >= k-1)
         // because each new subset must contain 'num' and k-1 elements from S[0...size-1]
         if (size >= k - 1) {
              num_new_subsets_formed = (k > 0) ? nCk_table[size][k - 1] : 0; // C(current size, k-1 needed)

              if (num_new_subsets_formed > 0) {
                  int combo_indices[k - 1]; // Indices into S[0...size-1]
                  for(int i = 0; i < k - 1; ++i) combo_indices[i] = i;

                  while(1) {
                      // Form the k-subset including 'num'
                      int current_subset[k];
                      for(int i = 0; i < k - 1; ++i) {
                         current_subset[i] = S[combo_indices[i]];
                      }
                      current_subset[k-1] = num; // Add the new number

                      // Calculate pattern and lookup rank
                      uint64 pat = numbers_to_pattern(current_subset, k);
                      int last_seen = lookup_subset(table, pat);
                      double rank = (last_seen >= 0) ? (double)(total_draws - last_seen - 1) : (double)total_draws;

                      // Update statistics for the *new* subsets
                      if (rank < min_rank_of_new_subsets) min_rank_of_new_subsets = rank;
                      sum_ranks_of_new_subsets += rank;

                      // Find next combination of k-1 indices from S[0...size-1]
                      int p = k - 2; // Indices 0 to k-2 for k-1 elements
                      while(p >= 0 && combo_indices[p] == size - (k - 1 - p)) {
                         p--;
                      }
                      if (p < 0) break; // No more combinations

                      combo_indices[p]++;
                      for(int x = p + 1; x < k - 1; ++x) {
                         combo_indices[x] = combo_indices[x - 1] + 1;
                      }
                  } // End while loop for iterating (k-1)-subsets
              } // End if num_new_subsets_formed > 0
         } // End if size >= k-1

         // Update overall stats for the NEXT level partial combo S[0...size]
         double next_min_rank = (current_min_rank < min_rank_of_new_subsets) ? current_min_rank : min_rank_of_new_subsets;
         double next_sum_ranks = sum_ranks_so_far + sum_ranks_of_new_subsets;
         // Total number of k-subsets evaluated up to S[0...size]
         uint64 Csk_plus_1 = (size + 1 >= k) ? nCk_table[size + 1][k] : 0;

         // --- PRUNING LOGIC ---
         int should_continue = 1; // Assume we continue by default
         if (list_is_full) {
             if (strcmp(m, "min") == 0) {
                 // Prune if the best possible min rank (next_min_rank) is already worse than the threshold
                 if (next_min_rank < threshold_primary ||
                    (next_min_rank == threshold_primary && (next_sum_ranks / Cjk) < threshold_secondary)) // Use projected avg for tiebreak (less accurate but ok)
                 {
                     should_continue = 0;
                 }
             } else { // m == "avg" - IMPROVED PRUNING
                 // Calculate the best possible average rank achievable from this point
                 // Lower bound: Assume all remaining subsets have rank 0 (best case)
                 double lower_bound_sum = next_sum_ranks; // Sum includes subsets evaluated so far
                 double lower_bound_avg = (Cjk > 0) ? (lower_bound_sum / (double)Cjk) : 0.0;

                 // Prune if the *best* possible average is already worse than the threshold
                 if (lower_bound_avg < threshold_primary ||
                    (lower_bound_avg == threshold_primary && next_min_rank < threshold_secondary)) // Use current min for tiebreak
                 {
                     should_continue = 0;
                 }

                 // --- Optional: Keep Upper Bound Check? ---
                 // An upper bound check might prune slightly differently in some cases,
                 // but the lower bound check is generally more effective for `avg`.
                 // Let's rely primarily on the lower bound check for `avg`.
                 /*
                 uint64 remaining_subsets = Cjk - Csk_plus_1;
                 double upper_bound_sum = next_sum_ranks + remaining_subsets * (double)total_draws;
                 double upper_bound_avg = (Cjk > 0) ? (upper_bound_sum / (double)Cjk) : 0.0;
                 // Original check: prune if NOT (upper_bound potentially good enough)
                 // Prune IF (upper_bound_avg < threshold_primary || (upper_bound_avg == threshold_primary && next_min_rank <= threshold_secondary))
                 if (upper_bound_avg < threshold_primary || (upper_bound_avg == threshold_primary && next_min_rank <= threshold_secondary)) {
                    // should_continue = 0; // This condition actually meant "prune" in the original logic's structure
                 }
                 */
             }
         } // End if list_is_full

         // Recurse if not pruned
         if (should_continue) {
              backtrack(S, size + 1, next_S_pattern, next_min_rank, next_sum_ranks, num + 1,
                        table, total_draws, max_number, j, k, Cjk, m, l,
                        thread_best, thread_filled);
         }
         // If pruned, the loop continues to the next 'num'

    } // End for loop iterating through 'num'
}


// --- Main Analysis Functions ---

static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data_all, // All draws available
    int use_count, // Number of draws to use for this analysis
    int j,
    int k,
    const char* m,
    int l,
    int n,
    int max_number,
    int* out_len // Pointer to store the number of results generated
) {
    *out_len = 0; // Initialize output length

    // 1. Build Subset Hash Table using the relevant 'use_count' draws
    SubsetTable* table = create_subset_table();
    if (!table) return NULL;
    // Process draws from index 0 up to use_count - 1
    for (int i = 0; i < use_count; i++) {
        // Pass pointer to the start of the i-th draw's data
        process_draw(&sorted_draws_data_all[i * 6], i, k, table);
    }

    // 2. Prepare for Parallel Backtracking
    int num_threads = omp_get_max_threads();
    if (num_threads <= 0) num_threads = 1; // Fallback for safety

    // Allocate space to hold the top L results for EACH thread
    ComboStats* all_threads_best = (ComboStats*)malloc(num_threads * l * sizeof(ComboStats));
    if (!all_threads_best) {
        free_subset_table(table);
        return NULL;
    }
     // Initialize all slots to invalid/empty state
    for(int i = 0; i < num_threads * l; ++i) {
        all_threads_best[i].len = 0; // Mark as invalid
        all_threads_best[i].avg_rank = -INFINITY;
        all_threads_best[i].min_rank = -INFINITY;
    }

    int global_error_occurred = 0; // Flag for critical errors in threads
    uint64 Cjk = (j >= k && k >= 0) ? nCk_table[j][k] : 0; // Calculate C(j,k)
     if (Cjk == 0 || Cjk == UINT64_MAX) { // Handle invalid combo (k>j) or overflow
         free(all_threads_best);
         free_subset_table(table);
         return NULL;
     }


    // 3. Parallel Backtracking Section
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int* S_local = (int*)malloc(j * sizeof(int)); // Thread-local combo buffer
        // Pointer to this thread's section in the shared results array
        ComboStats* thread_best_local = &all_threads_best[thread_id * l];
        int thread_filled_local = 0; // Local count for this thread's top L

        if (!S_local) {
            #pragma omp atomic write
            global_error_occurred = 1;
        } else {
            // Initialize this thread's local best list
            for (int i = 0; i < l; i++) {
                 thread_best_local[i].len = 0;
                 thread_best_local[i].avg_rank = -INFINITY;
                 thread_best_local[i].min_rank = -INFINITY;
            }

            // Distribute the starting number 'first_num' across threads
            #pragma omp for schedule(dynamic) nowait // Allow threads to proceed to merge earlier
            for (int first_num = 1; first_num <= max_number - j + 1; first_num++) {
                 if (global_error_occurred) continue; // Skip if error occurred elsewhere

                 S_local[0] = first_num;
                 uint64 current_S_pattern = (1ULL << (first_num - 1));
                 double current_min_rank = (double)(use_count + 1.0); // Worst possible rank + 1
                 double sum_ranks_so_far = 0.0;

                 // Start the recursive backtracking
                 backtrack(S_local, 1, current_S_pattern, current_min_rank, sum_ranks_so_far, first_num + 1,
                           table, use_count, max_number, j, k, Cjk, m, l,
                           thread_best_local, &thread_filled_local);
            } // End of parallel for loop

            free(S_local); // Free thread-local buffer
        } // End if S_local allocated successfully
    } // End of parallel region

    // Check for errors during parallel execution
    if (global_error_occurred) {
        free(all_threads_best);
        free_subset_table(table);
        return NULL;
    }

    // 4. Merge Results from All Threads
    ComboStats* final_best = (ComboStats*)malloc(l * sizeof(ComboStats));
     if (!final_best) {
         free(all_threads_best);
         free_subset_table(table);
         return NULL;
     }
     // Initialize final list
    for(int i = 0; i < l; ++i) {
         final_best[i].len = 0;
         final_best[i].avg_rank = -INFINITY;
         final_best[i].min_rank = -INFINITY;
    }
    int final_filled = 0;

    // Iterate through each result from each thread's list
    for (int i = 0; i < num_threads * l; ++i) {
        // Skip invalid/empty slots from threads
        if (all_threads_best[i].len != j) continue;

        // Compare and insert into the final_best list (same logic as in backtrack base case)
        int should_insert = 0;
        if (final_filled < l) {
            should_insert = 1;
        } else {
            if (strcmp(m, "avg") == 0) {
                should_insert = (all_threads_best[i].avg_rank > final_best[l - 1].avg_rank) ||
                                (all_threads_best[i].avg_rank == final_best[l - 1].avg_rank && all_threads_best[i].min_rank > final_best[l - 1].min_rank);
            } else { // m == "min"
                should_insert = (all_threads_best[i].min_rank > final_best[l - 1].min_rank) ||
                                (all_threads_best[i].min_rank == final_best[l - 1].min_rank && all_threads_best[i].avg_rank > final_best[l - 1].avg_rank);
            }
        }

        if (should_insert) {
            int insert_pos = l - 1;
            if (final_filled < l) {
                insert_pos = final_filled;
                final_filled++;
            }
            while (insert_pos > 0) {
                int compare_to_prev = 0;
                if (strcmp(m, "avg") == 0) {
                     compare_to_prev = (all_threads_best[i].avg_rank > final_best[insert_pos - 1].avg_rank) ||
                                      (all_threads_best[i].avg_rank == final_best[insert_pos - 1].avg_rank && all_threads_best[i].min_rank > final_best[insert_pos - 1].min_rank);
                } else { // m == "min"
                     compare_to_prev = (all_threads_best[i].min_rank > final_best[insert_pos - 1].min_rank) ||
                                      (all_threads_best[i].min_rank == final_best[insert_pos - 1].min_rank && all_threads_best[i].avg_rank > final_best[insert_pos - 1].avg_rank);
                }
                if (compare_to_prev) {
                    memcpy(&final_best[insert_pos], &final_best[insert_pos - 1], sizeof(ComboStats));
                    insert_pos--;
                } else {
                    break;
                }
            }
            if (insert_pos >= 0 && insert_pos < l) { // Should always be true here
                 memcpy(&final_best[insert_pos], &all_threads_best[i], sizeof(ComboStats));
            }
        }
    } // End merging loop

    free(all_threads_best); // Free the temporary thread results buffer

    // 5. Format Top-L Results
    // Allocate the final output structure array (size l + n)
    AnalysisResultItem* results = (AnalysisResultItem*)calloc(l + n, sizeof(AnalysisResultItem));
    if (!results) {
        free(final_best);
        free_subset_table(table);
        return NULL;
    }

    int results_count = 0;
    for (int i = 0; i < final_filled; i++) {
        if (results_count >= l + n) break; // Safety break
        format_combo(final_best[i].combo, final_best[i].len, results[results_count].combination);
        // Format subsets using the existing hash table
        format_subsets(final_best[i].combo, j, k, use_count, table, results[results_count].subsets);
        results[results_count].avg_rank = final_best[i].avg_rank;
        results[results_count].min_value = final_best[i].min_rank;
        results[results_count].is_chain_result = 0; // Mark as standard result
        // Chain-specific fields are not used here
        results[results_count].draw_offset = 0;
        results[results_count].draws_until_common = 0;
        results[results_count].analysis_start_draw = 0;
        results_count++;
    }

    // 6. Handle N non-overlapping combos (select from final_best)
    int second_table_start_index = results_count; // Where the N results will start
    int second_table_count = 0;
    if (n > 0 && final_filled > 0) {
        int* pick_indices = (int*)malloc(final_filled * sizeof(int));
        if (pick_indices) {
            memset(pick_indices, -1, final_filled * sizeof(int));
            int chosen = 0;
            pick_indices[chosen++] = 0; // Always pick the best one

            for (int i = 1; i < final_filled && chosen < n; i++) {
                uint64 pat_i = final_best[i].pattern;
                int overlap = 0;
                for (int c = 0; c < chosen; c++) {
                    int idxC = pick_indices[c];
                    uint64 pat_c = final_best[idxC].pattern;
                    // Check if intersection has k or more numbers
                    if (popcount64(pat_i & pat_c) >= k) {
                        overlap = 1;
                        break;
                    }
                }
                if (!overlap) {
                    pick_indices[chosen++] = i;
                }
            }
            second_table_count = chosen; // Actual number chosen

            // Format the N non-overlapping results (append to the main results array)
            for (int i = 0; i < second_table_count; i++) {
                 int current_result_index = second_table_start_index + i;
                 if (current_result_index >= l + n) break; // Don't exceed allocated size

                 int idx = pick_indices[i]; // Index in final_best array
                 format_combo(final_best[idx].combo, final_best[idx].len, results[current_result_index].combination);
                 format_subsets(final_best[idx].combo, j, k, use_count, table, results[current_result_index].subsets);
                 results[current_result_index].avg_rank = final_best[idx].avg_rank;
                 results[current_result_index].min_value = final_best[idx].min_rank;
                 results[current_result_index].is_chain_result = 0;
                 // Other fields remain 0 or unused for standard analysis N-results
            }
            results_count += second_table_count; // Update total count
            free(pick_indices);
        }
        // else: Allocation failed for pick_indices, skip N part
    }


    // 7. Final Cleanup and Return
    *out_len = results_count; // Set the actual number of items filled

    free_subset_table(table);
    free(final_best);

    if (results_count == 0) { // If no results were found at all
        free(results);
        return NULL;
    }

    return results;
}


static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data_all, // All draws
    int draws_count,            // Total number of draws
    int initial_offset,         // Starting offset
    int j,
    int k,
    const char* m,
    int max_number,
    int* out_len
) {
    *out_len = 0;
    // Allocate space for results - max possible chain length is draws_count
    AnalysisResultItem* chain_results = (AnalysisResultItem*)calloc(draws_count + 1, sizeof(AnalysisResultItem)); // Generous allocation
    if (!chain_results) return NULL;

    // Precompute bit patterns for all draws for faster overlap checking later
    uint64* draw_patterns = (uint64*)malloc(draws_count * sizeof(uint64));
    if (!draw_patterns) {
        free(chain_results);
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        draw_patterns[i] = numbers_to_pattern(&sorted_draws_data_all[i * 6], 6);
    }

    SubsetTable* table = create_subset_table(); // Create hash table once
    if (!table) {
        free(draw_patterns);
        free(chain_results);
        return NULL;
    }

    int chain_index = 0;
    int current_offset = initial_offset;
    uint64 Cjk = (j >= k && k >= 0) ? nCk_table[j][k] : 0;
     if (Cjk == 0 || Cjk == UINT64_MAX) { // Handle invalid combo (k>j) or overflow
         free(table);
         free(draw_patterns);
         free(chain_results);
         return NULL;
     }


    // Allocate buffer for backtrack (top-1 search)
    ComboStats best_chain_combo_buffer; // Find top-1
    int* S_chain = (int*)malloc(j * sizeof(int));
    if (!S_chain) {
         free(table);
         free(draw_patterns);
         free(chain_results);
         return NULL;
    }


    // --- Chain Loop ---
    while (1) {
        // Determine draws to use for this iteration
        int use_count = draws_count - current_offset;
        if (use_count < 1 || current_offset < 0) {
             break; // Stop if no draws left or offset is invalid
        }
        const int* current_draws_ptr = sorted_draws_data_all; // Analyze from the beginning up to use_count

        // 1. Prepare subset table for this iteration's draws
        clear_subset_table(table); // Reset hash table contents
        for (int i = 0; i < use_count; i++) {
            process_draw(&current_draws_ptr[i * 6], i, k, table);
        }

        // 2. Find the top-1 combination for this set of draws
        best_chain_combo_buffer.len = 0; // Reset best combo for this iteration
        best_chain_combo_buffer.avg_rank = -INFINITY;
        best_chain_combo_buffer.min_rank = -INFINITY;
        int filled_chain = 0;

        // Perform backtrack search for l=1 (could be optimized, but use existing function)
        // NOTE: This part is NOT parallelized as it's sequential chain logic.
        for (int first_num = 1; first_num <= max_number - j + 1; first_num++) {
             S_chain[0] = first_num;
             uint64 current_S_pattern = (1ULL << (first_num - 1));
             double current_min_rank = (double)(use_count + 1.0);
             double sum_ranks_so_far = 0.0;

             backtrack(S_chain, 1, current_S_pattern, current_min_rank, sum_ranks_so_far, first_num + 1,
                       table, use_count, max_number, j, k, Cjk, m, 1, // l=1
                       &best_chain_combo_buffer, &filled_chain);
        }


        // Check if a valid top-1 combo was found
        if (filled_chain == 0 || best_chain_combo_buffer.len != j) {
            break; // Stop chain if no combo found for this offset
        }

        // 3. Store the found top-1 combo results
        if (chain_index >= draws_count + 1) break; // Safety break if exceed allocation

        AnalysisResultItem* out_item = &chain_results[chain_index];
        format_combo(best_chain_combo_buffer.combo, best_chain_combo_buffer.len, out_item->combination);
        format_subsets(best_chain_combo_buffer.combo, j, k, use_count, table, out_item->subsets); // Use current table
        out_item->avg_rank = best_chain_combo_buffer.avg_rank;
        out_item->min_value = best_chain_combo_buffer.min_rank;
        out_item->is_chain_result = 1;
        out_item->draw_offset = chain_index + 1; // "Analysis #" field (1-based)
        out_item->analysis_start_draw = draws_count - current_offset; // "For Draw" field

        // 4. Find how many draws until a common k-subset appears (forward search)
        uint64 combo_pat = best_chain_combo_buffer.pattern;
        int steps_forward;
        int found_common = 0;
        // Search draws from (use_count) up to (draws_count - 1)
        // which corresponds to original indices from (draws_count - current_offset) to (draws_count - 1)
        for (steps_forward = 1; steps_forward <= current_offset; steps_forward++) {
            // Index in the original sorted_draws_data_all array
            int forward_draw_idx = (draws_count - current_offset) + steps_forward -1;
             if (forward_draw_idx >= draws_count) break; // Should not happen

            uint64 forward_draw_pat = draw_patterns[forward_draw_idx];
            // Check for k-subset overlap
            if (popcount64(combo_pat & forward_draw_pat) >= k) {
                found_common = 1;
                break; // Found overlap 'steps_forward' draws ahead
            }
        }
        // If loop finishes without break, steps_forward = current_offset + 1
        // Note: The duration is steps_forward - 1 (0 if found immediately)
        out_item->draws_until_common = (steps_forward > 0) ? (steps_forward - 1) : 0;

        // 5. Adjust offset for the next iteration
        current_offset -= steps_forward; // Decrease offset by steps taken

        chain_index++; // Move to the next result slot

    } // End while loop for chain

    // --- Cleanup for Chain Analysis ---
    free(S_chain);
    free_subset_table(table);
    free(draw_patterns);

    *out_len = chain_index; // Number of valid chain results generated
    if (chain_index == 0) {
        free(chain_results); // Free if no results
        return NULL;
    }

    // Optional: Trim the results array if significantly over-allocated?
    // AnalysisResultItem* trimmed_results = realloc(chain_results, chain_index * sizeof(AnalysisResultItem));
    // return (trimmed_results) ? trimmed_results : chain_results; // Return realloc'd or original if realloc fails
    return chain_results; // Return potentially over-allocated array for simplicity
}

// --- Main Entry Point & Cleanup ---

AnalysisResultItem* run_analysis_c(
    const char* game_type,
    int** draws, // Array of pointers to integer arrays
    int draws_count,
    int j,
    int k,
    const char* m,
    int l,
    int n,
    int last_offset,
    int* out_len // Output: number of results returned
) {
    *out_len = 0; // Default output length

    // Basic Parameter Validation
    if (!game_type || !draws || !m || !out_len || draws_count < 0 || j < 0 || k < 0 || l < -1 || n < 0) {
        return NULL; // Invalid input pointers or counts
    }
    if (j > MAX_ALLOWED_J || j < k || k < 1) {
         // fprintf(stderr, "Warning: Invalid j/k values (j=%d, k=%d). Must have 1 <= k <= j <= %d.\n", j, k, MAX_ALLOWED_J);
         return NULL; // Invalid j/k logic combination
    }
     if (l == 0 && n == 0) return NULL; // No results requested


    init_tables(); // Ensure nCk table is ready

    // Determine game parameters
    int max_number = (strstr(game_type, "6_49")) ? 49 : 42; // Default to 6/42 if not 6/49
    if (max_number <= 0 || max_number > MAX_NUMBERS) {
         // fprintf(stderr, "Warning: Invalid max_number deduced (%d). Using default 42.\n", max_number);
         max_number = 42; // Fallback
    }


    // Validate last_offset against draws_count
    if (last_offset < 0) last_offset = 0;
    // If offset is beyond the last draw, effectively use all draws (offset=0) or none?
    // Let's adjust it to be within bounds [0, draws_count].
    // If last_offset >= draws_count, it means we skip all draws. Let run_standard/chain handle use_count=0.
    if (last_offset > draws_count) last_offset = draws_count;


    // Prepare draws data: Copy and sort each draw into a contiguous block
    // This avoids issues with potentially non-contiguous `draws` input
    int* sorted_draws_data_all = (int*)malloc(draws_count * 6 * sizeof(int));
    if (!sorted_draws_data_all) return NULL;

    for (int i = 0; i < draws_count; i++) {
        int temp[6];
        // Check if draws[i] is valid before dereferencing
        if (!draws[i]) {
            free(sorted_draws_data_all);
            return NULL; // Invalid input row
        }
        // Copy numbers safely - handle potential NULLs in input? Assume valid integers for now.
        for (int z = 0; z < 6; z++) temp[z] = draws[i][z];

        // Sort the 6 numbers (simple insertion sort is fine for N=6)
        for(int ii = 1; ii < 6; ++ii) {
            int key = temp[ii];
            int jj = ii - 1;
            while(jj >= 0 && temp[jj] > key) {
                temp[jj + 1] = temp[jj];
                jj = jj - 1;
            }
            temp[jj + 1] = key;
        }
        // Copy sorted numbers to the contiguous block
        memcpy(&sorted_draws_data_all[i * 6], temp, 6 * sizeof(int));
    }

    // --- Dispatch to appropriate analysis function ---
    AnalysisResultItem* results = NULL;
    if (l != -1) { // Standard Analysis (Top-L + N non-overlapping)
        int use_count = draws_count - last_offset;
        if (use_count < k) { // Need at least k draws to form k-subsets
             // fprintf(stderr, "Warning: Not enough draws (%d after offset) for k=%d.\n", use_count, k);
             // Allow proceeding, backtrack will handle Cjk=0 if j<k
        }
        // Call standard analysis using draws from index 0 to use_count-1
        results = run_standard_analysis(
                      sorted_draws_data_all, // Pass pointer to all data
                      use_count,
                      j, k, m, l, n, max_number,
                      out_len); // out_len will be updated
    } else { // Chain Analysis (l == -1)
        // Pass all draws and the initial offset
        results = run_chain_analysis(
                      sorted_draws_data_all,
                      draws_count,
                      last_offset, // Pass the initial offset for the chain
                      j, k, m, max_number,
                      out_len); // out_len will be updated
    }

    // --- Cleanup ---
    free(sorted_draws_data_all); // Free the copied/sorted draw data

    // Return the results (or NULL if error/no results)
    // Note: 'out_len' should be correctly set by the called function
    return results;
}

void free_analysis_results(AnalysisResultItem* results) {
    if (results) {
        free(results);
    }
}

// --- Formatting Functions (with basic buffer overflow checks) ---
static void format_combo(const int* combo, int len, char* out) {
    if (!out) return;
    int pos = 0;
    int remaining = MAX_COMBO_STR; // Use defined constant

    for (int i = 0; i < len; i++) {
        if (remaining <= 1) break; // Need space for number and null terminator

        int written;
        if (i > 0) {
            if (remaining < 3) break; // Need space for ", " and number
            written = snprintf(out + pos, remaining, ", %d", combo[i]);
        } else {
            written = snprintf(out + pos, remaining, "%d", combo[i]);
        }

        if (written < 0 || written >= remaining) {
            pos = MAX_COMBO_STR - 1; // Go to end if error or exact fill
            break;
        }
        pos += written;
        remaining -= written;
    }
    out[pos] = '\0'; // Ensure null termination
}

static void format_subsets(const int* combo, int j, int k, int total_draws,
                          const SubsetTable* table, char* out) {
    if (!out) return;
    out[0] = '\0'; // Start with empty string

    // Basic validation
     if (k < 1 || k > j || j > MAX_NUMBERS) {
         snprintf(out, MAX_SUBSETS_STR, "[]");
         return;
     }

    uint64 exact_subset_count_64 = nCk_table[j][k];
     if (exact_subset_count_64 == 0 || exact_subset_count_64 == UINT64_MAX) {
         snprintf(out, MAX_SUBSETS_STR, "[]"); // Handle k>j or nCk overflow
         return;
     }
    // Prevent excessive memory allocation or processing if C(j, k) is huge
    const int MAX_SUBSETS_TO_PROCESS = 100000; // Limit processing/memory
    if (exact_subset_count_64 > MAX_SUBSETS_TO_PROCESS) {
         snprintf(out, MAX_SUBSETS_STR, "[Too many subsets C(%d,%d) to process/format]", j, k);
         return;
    }
    int exact_subset_count = (int)exact_subset_count_64;

    // --- Temporary structure for sorting subsets ---
    typedef struct {
        int numbers[MAX_NUMBERS]; // Store the k numbers
        int rank;
    } SubsetInfo;

    SubsetInfo* subsets = (SubsetInfo*)malloc(exact_subset_count * sizeof(SubsetInfo));
    if (!subsets) {
        snprintf(out, MAX_SUBSETS_STR, "[]"); // Allocation failed
        return;
    }
    int subset_count = 0; // Actual number generated

    // --- Generate all k-subsets of the j-combination 'combo' ---
    int idx[MAX_NUMBERS]; // Indices into 'combo' array
    for (int i = 0; i < k; i++) idx[i] = i;

    while (subset_count < exact_subset_count) {
        // Form the k-subset using indices into 'combo'
        int current_subset_nums[k];
        for (int i = 0; i < k; i++) {
            current_subset_nums[i] = combo[idx[i]];
        }
        memcpy(subsets[subset_count].numbers, current_subset_nums, k * sizeof(int));

        // Calculate pattern and rank
        uint64 pat = numbers_to_pattern(current_subset_nums, k);
        int last_seen = lookup_subset(table, pat);
        int rank = (last_seen >= 0) ? (total_draws - last_seen - 1) : total_draws;
        subsets[subset_count].rank = rank;
        subset_count++;

        // Generate next combination of indices
        int p = k - 1;
        while (p >= 0 && idx[p] == j - k + p) p--;
        if (p < 0) break; // Last combination generated
        idx[p]++;
        for (int x = p + 1; x < k; x++) idx[x] = idx[x - 1] + 1;
    }

    // --- Sort subsets by rank (descending) ---
    // Use qsort for potentially larger counts
    qsort(subsets, subset_count, sizeof(SubsetInfo),
         [](const void* a, const void* b) {
             int rankA = ((SubsetInfo*)a)->rank;
             int rankB = ((SubsetInfo*)b)->rank;
             if (rankB > rankA) return 1;  // Sort descending
             if (rankB < rankA) return -1;
             return 0;
         });


    // --- Format into string with buffer checks ---
    int pos = 0;
    int remaining = MAX_SUBSETS_STR;

    if (remaining > 1) { out[pos++] = '['; remaining--; }

    for (int i = 0; i < subset_count; i++) {
        if (remaining <= 1) break; // Need space for content and ']'

        // Add comma and space before second and subsequent items
        if (i > 0) {
            if (remaining < 3) break; // Need ", " + content
            out[pos++] = ','; out[pos++] = ' ';
            remaining -= 2;
        }

        // Format single subset: "((num1, ..., numk), rank)"
        if (remaining < 5) break; // Min needed: "((),)"
        out[pos++] = '('; out[pos++] = '(';
        remaining -= 2;

        // Format numbers
        for (int n = 0; n < k; n++) {
             if (remaining <= 1) break;
             int num_written;
             if (n > 0) {
                 if (remaining < 3) break; // Need ", " + digit
                 num_written = snprintf(out + pos, remaining, ", %d", subsets[i].numbers[n]);
             } else {
                 num_written = snprintf(out + pos, remaining, "%d", subsets[i].numbers[n]);
             }
             if (num_written < 0 || num_written >= remaining) { remaining = 0; break; }
             pos += num_written; remaining -= num_written;
        }
        if (remaining == 0) break;

        // Format rank part: "), rank)"
         if (remaining < 4) break; // Need "), )" + digit
         int rank_written = snprintf(out + pos, remaining, "), %d)", subsets[i].rank);
         if (rank_written < 0 || rank_written >= remaining) { remaining = 0; break; }
         pos += rank_written; remaining -= rank_written;

    } // End loop through subsets

    // Add closing bracket
    if (remaining > 0) {
        out[pos++] = ']';
    } else if (pos > 0) {
       // If buffer full, try to force ']' at the end if possible
       out[MAX_SUBSETS_STR - 2] = ']';
    }

    out[pos] = '\0'; // Ensure null termination

    free(subsets); // Free temporary subset array
}
