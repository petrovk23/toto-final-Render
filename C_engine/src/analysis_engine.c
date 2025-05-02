// C_engine/src/analysis_engine.c
// Modified to disable pruning based on comparing potential scores
// against the L-th element when m == "avg".

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
#define HASH_SIZE (1 << 26)  // 67M entries (Increased from original example)

typedef unsigned long long uint64;
typedef unsigned int uint32;

static uint64 nCk_table[MAX_NUMBERS][MAX_NUMBERS];
static int bit_count_table[256];
static int initialized = 0;

typedef struct {
    uint64* keys;
    int* values;
    int size;
    int capacity;
} SubsetTable;

typedef struct {
    uint64 pattern;
    double avg_rank;
    double min_rank;
    int combo[MAX_NUMBERS];
    int len;
} ComboStats;

static void init_tables();
static inline int popcount64(uint64 x);
static SubsetTable* create_subset_table(int max_entries);
static void free_subset_table(SubsetTable* table);
static inline uint32 hash_subset(uint64 pattern);
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value);
static inline int lookup_subset(const SubsetTable* table, uint64 pattern);
static inline uint64 numbers_to_pattern(const int* numbers, int count);
static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table);
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

static void backtrack(
    int* S,
    int size,
    uint64 current_S,
    double current_min_rank,
    double sum_current,
    int start_num,
    SubsetTable* table,
    int total_draws,
    int max_number,
    int j,
    int k,
    ComboStats* thread_best, // Array to store top L candidates for this thread
    int* thread_filled,      // Number of valid entries currently in thread_best
    int l,                   // Target number of top candidates
    const char* m,           // Sorting criterion ("avg" or "min")
    uint64 Cjk               // Total number of k-subsets for a j-combo (nCk(j,k))
);

static int compare_avg_rank(const void* a, const void* b);
static int compare_min_rank(const void* a, const void* b);

static void init_tables() {
    if (initialized) return;
    memset(nCk_table, 0, sizeof(nCk_table));
    for (int n = 0; n < MAX_NUMBERS; n++) {
        nCk_table[n][0] = 1;
        for (int k = 1; k <= n; k++) {
            // Prevent overflow for large n/k, though unlikely needed for MAX_NUMBERS=50
            if (nCk_table[n - 1][k - 1] > ULLONG_MAX - nCk_table[n - 1][k]) {
                 nCk_table[n][k] = ULLONG_MAX; // Or handle error appropriately
            } else {
                 nCk_table[n][k] = nCk_table[n - 1][k - 1] + nCk_table[n - 1][k];
            }
        }
    }
    // Ensure nCk(j,k) didn't overflow if needed
    // ...

    for (int i = 0; i < 256; i++) {
        int c = 0;
        for (int b = 0; b < 8; b++) {
            if (i & (1 << b)) c++;
        }
        bit_count_table[i] = c;
    }
    initialized = 1;
}

// Use GCC's builtin popcount for performance if available
#ifdef __GNUC__
static inline int popcount64(uint64 x) {
    return __builtin_popcountll(x);
}
#else
// Fallback popcount implementation
static inline int popcount64(uint64 x) {
    int count = 0;
    while (x > 0) {
        x &= (x - 1); // clear the least significant bit set
        count++;
    }
    return count;
    // Alternative using precomputed table (as originally present)
    // int res = 0;
    // for (int i = 0; i < 8; i++) {
    //     res += bit_count_table[x & 0xFF];
    //     x >>= 8;
    // }
    // return res;
}
#endif


static SubsetTable* create_subset_table(int max_entries) {
    SubsetTable* t = (SubsetTable*)malloc(sizeof(SubsetTable));
    if (!t) return NULL;
    t->size = 0; // Should not be used with hash table
    t->capacity = max_entries;
    // Use calloc to initialize keys to 0 and simplify hash probing logic
    t->keys = (uint64*)calloc(max_entries, sizeof(uint64));
    // Initialize values to -1 to distinguish empty slots from valid index 0
    t->values = (int*)malloc(max_entries * sizeof(int));
    if (!t->keys || !t->values) {
        free(t->keys);
        free(t->values);
        free(t);
        return NULL;
    }
    for (int i = 0; i < max_entries; i++) {
        t->values[i] = -1; // Sentinel for empty slot
    }
    return t;
}

static void free_subset_table(SubsetTable* table) {
    if (!table) return;
    free(table->keys);
    free(table->values);
    free(table);
}

// Simple hash function (consider alternatives like FNV or Murmur if needed)
// Using multiplication and XOR folding
static inline uint32 hash_subset(uint64 pattern) {
    pattern = (pattern ^ (pattern >> 30)) * 0xbf58476d1ce4e5b9ULL;
    pattern = (pattern ^ (pattern >> 27)) * 0x94d049bb133111ebULL;
    pattern = pattern ^ (pattern >> 31);
    return (uint32)(pattern & (HASH_SIZE - 1)); // Mask for table size
}


// Linear probing hash table insertion
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value) {
    uint32 idx = hash_subset(pattern);
    uint32 start_idx = idx;
    while (1) {
        // If slot is empty (key is 0) or key matches, insert/update
        if (table->keys[idx] == 0 || table->keys[idx] == pattern) {
            table->keys[idx] = pattern;
            table->values[idx] = value;
            return;
        }
        // Move to next slot, wrapping around
        idx = (idx + 1) & (HASH_SIZE - 1);
        // If we wrapped around back to start, table is full (should not happen with good sizing)
        if (idx == start_idx) {
            fprintf(stderr, "Error: Hash table full! Cannot insert pattern %llu\n", pattern);
             // Handle error: maybe rehash or exit
            exit(EXIT_FAILURE);
            return;
        }
    }
}

// Linear probing hash table lookup
static inline int lookup_subset(const SubsetTable* table, uint64 pattern) {
    uint32 idx = hash_subset(pattern);
    uint32 start_idx = idx;
    while (1) {
        // If we find the key, return the value
        if (table->keys[idx] == pattern) {
            return table->values[idx];
        }
        // If we hit an empty slot (key is 0), the pattern is not in the table
        if (table->keys[idx] == 0) {
            return -1; // Not found
        }
        // Move to next slot, wrapping around
        idx = (idx + 1) & (HASH_SIZE - 1);
        // If we wrapped around back to start without finding, it's not there
        if (idx == start_idx) {
            return -1; // Not found
        }
    }
}

static inline uint64 numbers_to_pattern(const int* numbers, int count) {
    uint64 p = 0ULL;
    for (int i = 0; i < count; i++) {
        if (numbers[i] > 0 && numbers[i] <= 64) { // Basic check
             p |= (1ULL << (numbers[i] - 1));
        } else {
             fprintf(stderr, "Warning: Invalid number %d encountered in numbers_to_pattern.\n", numbers[i]);
        }
    }
    return p;
}

// Process a single draw: find all its k-subsets and add/update them in the table
static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table) {
    if (k > 6 || k < 1) return; // k must be within bounds of the draw size

    int indices[6]; // Indices to select k elements from the 6 numbers in draw
    for (int i = 0; i < k; i++) {
        indices[i] = i; // Start with the first k indices {0, 1, ..., k-1}
    }

    // Iterate through all combinations of k indices from 0 to 5
    while (1) {
        uint64 subset_pattern = 0ULL;
        int subset_numbers[k]; // Temporary storage if needed, mainly for pattern
        int valid_subset = 1;

        // Build the pattern for the current subset
        for (int i = 0; i < k; i++) {
             int num = draw[indices[i]];
             if (num > 0 && num <= 64) {
                 subset_pattern |= (1ULL << (num - 1));
                 subset_numbers[i] = num; // Store if needed
             } else {
                 fprintf(stderr, "Warning: Invalid number %d in draw at index %d.\n", num, draw_idx);
                 valid_subset = 0;
                 break; // Skip this subset if number is invalid
             }
        }

        // Insert the pattern into the hash table if valid
        if (valid_subset) {
            insert_subset(table, subset_pattern, draw_idx);
        }

        // Find the next combination of indices (lexicographical order)
        int pos_to_increment = k - 1;
        // Find rightmost index that can be incremented
        while (pos_to_increment >= 0 && indices[pos_to_increment] == 6 - (k - pos_to_increment)) {
            pos_to_increment--;
        }

        // If no index can be incremented, we're done with all combinations
        if (pos_to_increment < 0) {
            break;
        }

        // Increment the found index
        indices[pos_to_increment]++;

        // Reset subsequent indices
        for (int i = pos_to_increment + 1; i < k; i++) {
            indices[i] = indices[i - 1] + 1;
        }
    }
}

// The core backtracking function to explore combinations
static void backtrack(
    int* S,                  // Current partial combination being built
    int size,                // Current size of S (number of elements added so far)
    uint64 current_S_pattern,// Bit pattern of the current partial combination S
    double current_min_rank, // Minimum rank found among k-subsets formed ONLY from S
    double sum_rank_subsets, // Sum of ranks of k-subsets formed ONLY from S
    int start_num,           // Next number to consider adding to S
    SubsetTable* table,      // Hash table mapping k-subset patterns to last seen draw index
    int total_draws,         // Total number of draws being considered for ranking
    int max_number,          // Maximum allowed number in a combination (e.g., 42 or 49)
    int j,                   // Target size of the final combination
    int k,                   // Size of subsets to calculate ranks for
    ComboStats* thread_best, // Array to store top L candidates for this thread
    int* thread_filled,      // Number of valid entries currently in thread_best
    int l,                   // Target number of top candidates
    const char* m,           // Sorting criterion ("avg" or "min")
    uint64 Cjk               // Total number of k-subsets for a j-combo (nCk(j,k))
) {
    // Base Case: Combination S has reached the target size j
    if (size == j) {
        // Calculate final stats for the completed combination S
        double final_avg_rank = (Cjk > 0) ? (sum_rank_subsets / (double)Cjk) : 0.0;
        double final_min_rank = current_min_rank; // Min rank doesn't change at the last step

        int should_insert = 0;
        // Check if this completed combination should be inserted into thread_best
        if (*thread_filled < l) {
            // If the list is not full, always insert
            should_insert = 1;
        } else {
            // If the list is full, compare with the worst element (at index l-1)
            if (strcmp(m, "avg") == 0) {
                // Compare based on avg_rank first, then min_rank for ties
                should_insert = (final_avg_rank > thread_best[l - 1].avg_rank) ||
                                (final_avg_rank == thread_best[l - 1].avg_rank && final_min_rank > thread_best[l - 1].min_rank);
            } else { // m == "min"
                // Compare based on min_rank first, then avg_rank for ties
                should_insert = (final_min_rank > thread_best[l - 1].min_rank) ||
                                (final_min_rank == thread_best[l - 1].min_rank && final_avg_rank > thread_best[l - 1].avg_rank);
            }
        }

        // If it should be inserted, add it and maintain sorted order
        if (should_insert) {
            int insert_pos = l - 1; // Position to insert (initially the last slot)
            if (*thread_filled < l) {
                insert_pos = *thread_filled; // Insert at the end if not full
                (*thread_filled)++;          // Increment count
            }

            // Shift elements down if necessary to make space
            for (int i = l - 1; i > insert_pos; i--) {
                 thread_best[i] = thread_best[i - 1];
            }

            // Store the new best combination's data
            for (int i = 0; i < j; i++) thread_best[insert_pos].combo[i] = S[i];
            thread_best[insert_pos].len = j;
            thread_best[insert_pos].avg_rank = final_avg_rank;
            thread_best[insert_pos].min_rank = final_min_rank;
            thread_best[insert_pos].pattern = current_S_pattern; // Store the final pattern

            // Bubble the newly inserted element up to its correct sorted position
            for (int i = insert_pos; i > 0; i--) {
                int should_swap = 0;
                if (strcmp(m, "avg") == 0) {
                    should_swap = (thread_best[i].avg_rank > thread_best[i - 1].avg_rank) ||
                                  (thread_best[i].avg_rank == thread_best[i - 1].avg_rank &&
                                   thread_best[i].min_rank > thread_best[i - 1].min_rank);
                } else { // m == "min"
                    should_swap = (thread_best[i].min_rank > thread_best[i - 1].min_rank) ||
                                  (thread_best[i].min_rank == thread_best[i - 1].min_rank &&
                                   thread_best[i].avg_rank > thread_best[i - 1].avg_rank);
                }

                if (should_swap) {
                    ComboStats tmp = thread_best[i];
                    thread_best[i] = thread_best[i - 1];
                    thread_best[i - 1] = tmp;
                } else {
                    break; // In correct position relative to the element before it
                }
            }
        }
        return; // End recursion for this path
    }

    // Recursive Step: Try adding each possible number 'num' from 'start_num' up to 'max_number'
    // Ensure there's enough remaining numbers to reach size j
    for (int num = start_num; num <= max_number - (j - size - 1); num++) {
        // Add the number 'num' to the current partial combination S
        S[size] = num;
        uint64 new_S_pattern = current_S_pattern | (1ULL << (num - 1));

        // Calculate stats for the *new* k-subsets formed by adding 'num'
        double min_rank_of_new_subsets = (double)(total_draws + 1.0); // Initialize high
        double sum_rank_of_new_subsets = 0.0;
        uint64 count_new_subsets = 0; // Should be nCk(size, k-1)

        // Only calculate subset ranks if we have enough elements to form a k-subset
        if (size >= k - 1) {
             count_new_subsets = nCk_table[size][k - 1];
             int indices[k - 1]; // Indices to choose k-1 elements from S[0...size-1]
             for (int i = 0; i < k - 1; i++) indices[i] = i;

             // Iterate through all combinations of k-1 indices from the existing S
             while (1) {
                 int subset_nums[k]; // The k-subset: k-1 from S + the new 'num'
                 uint64 subset_pattern = (1ULL << (num - 1)); // Start with the new number's bit

                 // Build the subset and its pattern
                 for (int i = 0; i < k - 1; i++) {
                     int prev_num = S[indices[i]];
                     subset_nums[i] = prev_num;
                     subset_pattern |= (1ULL << (prev_num - 1));
                 }
                 subset_nums[k - 1] = num; // Add the new number

                 // Lookup the rank of this new k-subset
                 int last_seen = lookup_subset(table, subset_pattern);
                 double rank = (last_seen >= 0) ? (double)(total_draws - last_seen - 1) : (double)total_draws;

                 // Update statistics based on this new subset's rank
                 if (rank < min_rank_of_new_subsets) min_rank_of_new_subsets = rank;
                 sum_rank_of_new_subsets += rank;

                 // Find the next combination of k-1 indices
                 int p = k - 2; // k-1 indices range from 0 to k-2
                 while (p >= 0 && indices[p] == size - (k - 1 - p)) {
                     p--;
                 }
                 if (p < 0) break; // No more combinations
                 indices[p]++;
                 for (int x = p + 1; x < k - 1; x++) {
                     indices[x] = indices[x - 1] + 1;
                 }
             }
        } else {
             // Not enough elements yet to form k-subsets including 'num'
             min_rank_of_new_subsets = (double)total_draws; // Default rank is max
             sum_rank_of_new_subsets = 0.0;
             count_new_subsets = 0;
        }

        // Update the overall min_rank and sum_rank for the new partial combination S + num
        double next_min_rank = (current_min_rank < min_rank_of_new_subsets) ? current_min_rank : min_rank_of_new_subsets;
        double next_sum_rank = sum_rank_subsets + sum_rank_of_new_subsets;

        // --- Pruning Logic ---
        int should_continue_path = 1; // Assume we continue by default

        // Check for pruning ONLY if the top-L list is full
        if (*thread_filled >= l) {
            // Apply pruning ONLY if m is "min"
            if (strcmp(m, "min") == 0) {
                // Calculate potential average rank upper bound (needed for tie-breaking)
                // Number of k-subsets formed so far = nCk(size+1, k)
                uint64 subsets_formed_count = (size + 1 >= k) ? nCk_table[size + 1][k] : 0;
                double upper_avg_rank_potential = (Cjk > 0) ?
                    (next_sum_rank + (Cjk - subsets_formed_count) * (double)total_draws) / (double)Cjk
                    : 0.0; // Best possible avg assuming remaining subsets get max rank

                // Determine if this path should be pruned based on 'min' criteria.
                // Prune if it's IMPOSSIBLE for this path to be better than thread_best[l-1].
                // Condition to *continue*: (potential_min > worst_min) OR (potential_min == worst_min AND potential_avg > worst_avg)
                // We set should_continue_path = 0 (prune) if the *negation* is true.
                if (! ( (next_min_rank > thread_best[l - 1].min_rank) ||
                         (next_min_rank == thread_best[l - 1].min_rank && upper_avg_rank_potential > thread_best[l - 1].avg_rank)
                       ) )
                {
                    should_continue_path = 0; // Prune this path
                }
            }
            // If m is "avg", should_continue_path remains 1 (no pruning based on comparison to thread_best)
        }
        // If the top-L list is not full (*thread_filled < l), should_continue_path also remains 1.

        // Make the recursive call ONLY if the path should continue (i.e., not pruned)
        if (should_continue_path) {
            backtrack(S, size + 1, new_S_pattern, next_min_rank, next_sum_rank, num + 1, table, total_draws, max_number, j, k, thread_best, thread_filled, l, m, Cjk);
        }
    }
}


// Comparison function for sorting final results by average rank (descending)
static int compare_avg_rank(const void* a, const void* b) {
    ComboStats* ca = (ComboStats*)a;
    ComboStats* cb = (ComboStats*)b;
    if (ca->avg_rank > cb->avg_rank) return -1; // Higher avg_rank comes first
    if (ca->avg_rank < cb->avg_rank) return 1;
    // Tie-breaker: higher min_rank comes first
    if (ca->min_rank > cb->min_rank) return -1;
    if (ca->min_rank < cb->min_rank) return 1;
    return 0; // Should ideally compare combo numbers for deterministic sort if all else equal
}

// Comparison function for sorting final results by minimum rank (descending)
static int compare_min_rank(const void* a, const void* b) {
    ComboStats* ca = (ComboStats*)a;
    ComboStats* cb = (ComboStats*)b;
    if (ca->min_rank > cb->min_rank) return -1; // Higher min_rank comes first
    if (ca->min_rank < cb->min_rank) return 1;
    // Tie-breaker: higher avg_rank comes first
    if (ca->avg_rank > cb->avg_rank) return -1;
    if (ca->avg_rank < cb->avg_rank) return 1;
    return 0; // Should ideally compare combo numbers for deterministic sort if all else equal
}

// Function to run the standard analysis (top L, optional N non-overlapping)
static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data, // Pointer to all draws (sorted internally), oldest first
    int use_count,           // Number of newest draws to use for analysis
    int j, int k,
    const char* m,
    int l, int n,             // l = top combos, n = non-overlapping combos
    int max_number,
    int* out_len              // Output: number of results returned
) {
    // 1. Build the subset table using the relevant draws
    // We use the 'use_count' newest draws. Their indices in sorted_draws_data
    // are from (total_draws - use_count) to (total_draws - 1).
    // However, the ranking uses relative indices from 0 to use_count-1.
    SubsetTable* table = create_subset_table(HASH_SIZE);
    if (!table) { *out_len = 0; return NULL; }

    int total_draws_available = 0; // Need total count if using offsets... assume use_count is the total for now
    // Let's assume sorted_draws_data ONLY contains the use_count draws needed.
    for (int i = 0; i < use_count; i++) {
        // Process draw with index 'i' relative to the start of the considered window
        process_draw(&sorted_draws_data[i * 6], i, k, table);
    }

    // 2. Prepare for parallel backtracking search
    int num_threads = omp_get_max_threads();
    // Allocate space to store the top L results from EACH thread
    ComboStats* all_threads_best = (ComboStats*)malloc(num_threads * l * sizeof(ComboStats));
    if (!all_threads_best) {
        free_subset_table(table);
        *out_len = 0;
        return NULL;
    }
    // Initialize ranks to a value indicating 'empty' or 'invalid'
    for (int i = 0; i < num_threads * l; ++i) {
         all_threads_best[i].avg_rank = -1.0; // Or some other indicator
         all_threads_best[i].min_rank = -1.0;
         all_threads_best[i].len = 0;
    }


    int global_error_occurred = 0; // Flag for critical errors within threads
    uint64 Cjk = (j >= k) ? nCk_table[j][k] : 0; // nCk(j,k)
    if (Cjk == 0 && j >= k) {
         fprintf(stderr, "Warning: C(%d, %d) calculation resulted in 0 or overflowed.\n", j, k);
         // Handle this? Maybe proceed if Cjk is just 0, error if overflow?
    }


    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int* S = (int*)malloc(j * sizeof(int)); // Thread-local combination buffer
        // Point to the section of the global array for this thread's results
        ComboStats* thread_best = &all_threads_best[thread_id * l];
        int thread_filled = 0; // Local count for this thread's results

        if (!S) {
            #pragma omp atomic write
            global_error_occurred = 1;
        } else {
            // Initialize this thread's best list (redundant due to main thread init, but safe)
             for (int i = 0; i < l; i++) {
                thread_best[i].avg_rank = -1.0;
                thread_best[i].min_rank = -1.0;
                thread_best[i].len = 0;
             }

            // Parallelize the outer loop of backtracking (by the first number)
            #pragma omp for schedule(dynamic) nowait // Use nowait if no implicit barrier needed
            for (int first_num = 1; first_num <= max_number - j + 1; first_num++) {
                if (!global_error_occurred) { // Check flag before starting work
                    S[0] = first_num;
                    uint64 current_S_pattern = (1ULL << (first_num - 1));
                    // Initial stats before adding the first number (effectively zero)
                    double current_min_rank = (double)(use_count + 1.0); // Worse than any possible rank
                    double sum_current = 0.0;

                    // Start the recursive backtracking search for this first number
                    backtrack(S, 1, current_S_pattern, current_min_rank, sum_current, first_num + 1,
                              table, use_count, max_number, j, k,
                              thread_best, &thread_filled, l, m, Cjk);
                }
            }
            free(S); // Free thread-local buffer
        }
    } // End of parallel region

    // Check if any thread encountered a critical error
    if (global_error_occurred) {
        free(all_threads_best);
        free_subset_table(table);
        *out_len = 0;
        return NULL;
    }

    // 3. Merge results from all threads
    // Collect all valid candidates found by all threads
    int total_candidates_found = 0;
    for (int i = 0; i < num_threads * l; ++i) {
        if (all_threads_best[i].len == j) { // Check if it's a valid, filled entry
            total_candidates_found++;
        }
    }

    if (total_candidates_found == 0) {
         free(all_threads_best);
         free_subset_table(table);
         *out_len = 0;
         return NULL;
    }


    ComboStats* final_candidates = (ComboStats*)malloc(total_candidates_found * sizeof(ComboStats));
     if (!final_candidates) {
        free(all_threads_best);
        free_subset_table(table);
        *out_len = 0;
        return NULL;
    }

    int current_candidate_idx = 0;
    for (int i = 0; i < num_threads * l; ++i) {
        if (all_threads_best[i].len == j) {
            final_candidates[current_candidate_idx++] = all_threads_best[i];
        }
    }
    free(all_threads_best); // Free the per-thread storage


    // Sort the collected candidates based on the chosen criterion 'm'
    if (strcmp(m, "avg") == 0) {
        qsort(final_candidates, total_candidates_found, sizeof(ComboStats), compare_avg_rank);
    } else { // m == "min"
        qsort(final_candidates, total_candidates_found, sizeof(ComboStats), compare_min_rank);
    }

    // 4. Select the final top L results
    int top_l_count = (total_candidates_found < l) ? total_candidates_found : l;
    // We need to store these best L stats temporarily if we need them for N calculation
    ComboStats* best_l_stats = (ComboStats*)malloc(top_l_count * sizeof(ComboStats));
     if (!best_l_stats) {
         free(final_candidates);
         free_subset_table(table);
         *out_len = 0;
         return NULL;
    }
    for (int i = 0; i < top_l_count; i++) {
        best_l_stats[i] = final_candidates[i];
    }


    // 5. Prepare the final output structure (AnalysisResultItem array)
    // Total potential size is L (top) + N (non-overlapping selected from top L)
    AnalysisResultItem* final_results = (AnalysisResultItem*)calloc(l + n, sizeof(AnalysisResultItem));
    if (!final_results) {
        free(best_l_stats);
        free(final_candidates);
        free_subset_table(table);
        *out_len = 0;
        return NULL;
    }

    int results_filled_count = 0;

    // Rebuild the subset table ONE MORE TIME to format the subset strings correctly for the final results
    // (This is slightly inefficient but simpler than passing subset info through the merge process)
    free_subset_table(table);
    table = create_subset_table(HASH_SIZE);
    if (!table) {
         free(best_l_stats);
         free(final_candidates);
         free(final_results);
         *out_len = 0;
         return NULL;
    }
    for (int i = 0; i < use_count; i++) {
        process_draw(&sorted_draws_data[i * 6], i, k, table);
    }


    // Fill the top L results into the output array
    for (int i = 0; i < top_l_count; i++) {
        format_combo(best_l_stats[i].combo, best_l_stats[i].len, final_results[results_filled_count].combination);
        format_subsets(best_l_stats[i].combo, j, k, use_count, table, final_results[results_filled_count].subsets);
        final_results[results_filled_count].avg_rank = best_l_stats[i].avg_rank;
        final_results[results_filled_count].min_value = best_l_stats[i].min_rank;
        final_results[results_filled_count].is_chain_result = 0; // Mark as standard result
         // Zero out chain-specific fields
        final_results[results_filled_count].draw_offset = 0;
        final_results[results_filled_count].draws_until_common = 0;
        final_results[results_filled_count].analysis_start_draw = 0;

        results_filled_count++;
    }

    // 6. Select N non-overlapping combinations if requested (n > 0)
    int non_overlapping_count = 0;
    if (n > 0 && top_l_count > 0) {
        int* selected_indices = (int*)malloc(top_l_count * sizeof(int)); // Store indices of selected non-overlapping combos from best_l_stats
         if (!selected_indices) {
             // Continue without non-overlapping, or handle error
             fprintf(stderr, "Warning: Failed to allocate memory for non-overlapping selection.\n");
         } else {
            memset(selected_indices, -1, top_l_count * sizeof(int)); // Init indices to -1
            int current_selection_count = 0;

            // Greedily select from the sorted top_l_stats
            for (int i = 0; i < top_l_count && current_selection_count < n; i++) {
                uint64 pattern_i = best_l_stats[i].pattern;
                int has_overlap = 0;
                // Check against previously selected combinations
                for (int sel_idx = 0; sel_idx < current_selection_count; sel_idx++) {
                    int chosen_combo_idx = selected_indices[sel_idx];
                    uint64 pattern_chosen = best_l_stats[chosen_combo_idx].pattern;
                    // Check if the intersection has at least k elements
                    uint64 intersection = pattern_i & pattern_chosen;
                    if (popcount64(intersection) >= k) {
                        has_overlap = 1;
                        break; // Overlaps with a previously selected combo
                    }
                }

                // If no overlap was found, select this combination
                if (!has_overlap) {
                    selected_indices[current_selection_count++] = i; // Store the index i
                }
            }
            non_overlapping_count = current_selection_count;


            // Add the selected non-overlapping results to the final_results array
            // Start filling from the position after the top L results
            int non_overlap_start_index = results_filled_count;
            for (int i = 0; i < non_overlapping_count; i++) {
                int combo_idx = selected_indices[i]; // Get the index from best_l_stats

                // Format and store this selected non-overlapping combo
                format_combo(best_l_stats[combo_idx].combo, best_l_stats[combo_idx].len, final_results[non_overlap_start_index + i].combination);
                format_subsets(best_l_stats[combo_idx].combo, j, k, use_count, table, final_results[non_overlap_start_index + i].subsets);
                final_results[non_overlap_start_index + i].avg_rank = best_l_stats[combo_idx].avg_rank;
                final_results[non_overlap_start_index + i].min_value = best_l_stats[combo_idx].min_rank;
                final_results[non_overlap_start_index + i].is_chain_result = 0; // Mark as standard result
                 // Zero out chain-specific fields
                final_results[non_overlap_start_index + i].draw_offset = 0;
                final_results[non_overlap_start_index + i].draws_until_common = 0;
                final_results[non_overlap_start_index + i].analysis_start_draw = 0;

                results_filled_count++; // Increment total results count
            }

             free(selected_indices);
         } // end else (!selected_indices)

    } // end if (n > 0 && top_l_count > 0)


    // Clean up intermediate allocations
    free(best_l_stats);
    free(final_candidates);
    free_subset_table(table);

    // Set the final output length and return the results
    *out_len = results_filled_count;
    if (results_filled_count == 0) {
        free(final_results); // Free if no results were actually added
        return NULL;
    }
    return final_results;
}


// Function to run the chain analysis (repeatedly finding top-1)
static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data, // All draws, sorted internally, oldest first
    int draws_count,         // Total number of draws available
    int initial_offset,      // Starting offset from the newest draw
    int j, int k,
    const char* m,
    int max_number,
    int* out_len              // Output: number of results in the chain
) {
    // Allocate space for chain results (max possible length = initial_offset + 1, add buffer)
    AnalysisResultItem* chain_results = (AnalysisResultItem*)calloc(initial_offset + 2, sizeof(AnalysisResultItem));
    if (!chain_results) {
        *out_len = 0;
        return NULL;
    }

    // Precompute bit patterns for all draws to speed up overlap checks
    uint64* draw_patterns = (uint64*)malloc(draws_count * sizeof(uint64));
    if (!draw_patterns) {
        free(chain_results);
        *out_len = 0;
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        draw_patterns[i] = numbers_to_pattern(&sorted_draws_data[i * 6], 6);
    }

    int chain_results_count = 0;
    int current_offset = initial_offset; // Offset from the END (newest draw)
    uint64 Cjk = (j >= k) ? nCk_table[j][k] : 0;

    // Loop until offset becomes invalid or no more results found
    while (current_offset >= 0 && current_offset < draws_count) {
        int use_count = draws_count - current_offset; // Number of draws to consider for this iteration
        if (use_count < 1) break; // Not enough draws

        // --- Find the Top-1 Combo for this offset ---
        // Build subset table for the current window of draws
        SubsetTable* table = create_subset_table(HASH_SIZE);
        if (!table) break; // Allocation error
        // Process draws from index 0 up to use_count-1 (relative indices)
        // These correspond to original draws from index 0 to draws_count - current_offset - 1
         int analysis_start_draw_index = 0; // The oldest draw used in this iteration
         for (int i = 0; i < use_count; i++) {
            process_draw(&sorted_draws_data[i * 6], i, k, table);
         }


        // Find the single best combination (l=1) using backtracking
        ComboStats best_combo_for_offset;
        memset(&best_combo_for_offset, 0, sizeof(ComboStats)); // Clear stats
        best_combo_for_offset.avg_rank = -1.0;
        best_combo_for_offset.min_rank = -1.0;
        best_combo_for_offset.len = 0;

        int filled_count = 0; // Will be 0 or 1
        int* S_chain = (int*)malloc(j * sizeof(int));

        if (!S_chain) {
            free_subset_table(table);
            break; // Allocation error
        }

        // We need to run the backtrack search. Since l=1, no need for parallel merge.
        // Can we simplify? Just find the best without OpenMP?
        // Let's use the same backtrack but with l=1 and no parallelism needed.

        // Start search from the first possible number
        for (int first_num = 1; first_num <= max_number - j + 1; first_num++) {
             S_chain[0] = first_num;
             uint64 current_S_pattern = (1ULL << (first_num - 1));
             double current_min_rank = (double)(use_count + 1.0);
             double sum_current = 0.0;

             backtrack(S_chain, 1, current_S_pattern, current_min_rank, sum_current, first_num + 1,
                       table, use_count, max_number, j, k,
                       &best_combo_for_offset, &filled_count, 1, m, Cjk); // l=1
        }
        free(S_chain);


        // If no combo was found (e.g., use_count too small?), stop the chain
        if (filled_count == 0 || best_combo_for_offset.len != j) {
            free_subset_table(table);
            break;
        }

        // --- Store the found Top-1 Combo ---
        AnalysisResultItem* current_result_item = &chain_results[chain_results_count];

        format_combo(best_combo_for_offset.combo, best_combo_for_offset.len, current_result_item->combination);
        // Format subsets using the same table
        format_subsets(best_combo_for_offset.combo, j, k, use_count, table, current_result_item->subsets);

        current_result_item->avg_rank = best_combo_for_offset.avg_rank;
        current_result_item->min_value = best_combo_for_offset.min_rank;
        current_result_item->is_chain_result = 1; // Mark as chain result
        current_result_item->draw_offset = chain_results_count + 1; // "Analysis #" (1-based)
        // "For Draw" = the index of the *newest* draw used in this analysis step (1-based)
        // Newest draw used is at index use_count-1 relative to start, which is draws_count - current_offset - 1 globally
        // The display might expect the draw number from the DB? Let's use the global index + 1.
        // The Python code used: 'Analysis Start Draw': row['draw_offset'] which seems wrong.
        // Let's use the original Python logic's apparent intent: 'Analysis Start Draw': draws_count - current_offset
        // The C struct uses analysis_start_draw.
        current_result_item->analysis_start_draw = draws_count - current_offset; // "For Draw" (index of newest draw used + 1?) Check Python template. Python uses 1-based 'Draw'. Let's use this.

        free_subset_table(table); // Done with table for this iteration

        // --- Find Draws Until Common Subset ---
        uint64 top_combo_pattern = best_combo_for_offset.pattern;
        int draws_forward = 0; // How many draws forward until overlap
        int common_found = 0;

        // Search draws *after* the current window used for analysis
        // Start checking from the draw immediately following the newest one used (index draws_count - current_offset)
        for (int i = 0; i < current_offset; i++) {
             int future_draw_index = (draws_count - current_offset) + i;
             if (future_draw_index >= draws_count) break; // Should not happen with loop limit

             uint64 future_draw_pat = draw_patterns[future_draw_index];
             uint64 intersection = top_combo_pattern & future_draw_pat;

             if (popcount64(intersection) >= k) {
                 common_found = 1;
                 draws_forward = i + 1; // Found after i+1 steps forward
                 break;
             }
        }

        // If no common subset found within remaining draws, calculate effective duration
        if (!common_found) {
            draws_forward = current_offset + 1; // Effective duration is all remaining + 1 more step
        }

        current_result_item->draws_until_common = draws_forward -1; // Duration is steps - 1

        // Update offset for the next iteration
        current_offset -= draws_forward; // Move offset back by the duration
        chain_results_count++;          // Increment count of results found

        // Check if we've exceeded allocated space (safety)
        if (chain_results_count >= initial_offset + 2) {
            fprintf(stderr, "Warning: Chain analysis exceeded allocated buffer size.\n");
            break;
        }

    } // End while loop

    free(draw_patterns);
    *out_len = chain_results_count;

    if (chain_results_count == 0) {
        free(chain_results);
        return NULL;
    }
    // Consider reallocating chain_results to exact size if memory is critical
    // AnalysisResultItem* final_chain_results = realloc(chain_results, chain_results_count * sizeof(AnalysisResultItem));
    // return final_chain_results ? final_chain_results : chain_results; // Return original on realloc failure

    return chain_results;
}


// Main entry point called from Python via ctypes
AnalysisResultItem* run_analysis_c(
    const char* game_type,    // e.g., "6_42", "6_49"
    int** draws,              // Array of pointers to draw arrays (int[6])
    int draws_count,          // Total number of draws provided
    int j, int k,
    const char* m,            // "avg" or "min"
    int l,                    // Top L combos, or -1 for chain analysis
    int n,                    // Number of non-overlapping combos (for l != -1)
    int last_offset,          // Offset from the last draw (0 means use all)
    int* out_len              // Output parameter: number of results returned
) {
    *out_len = 0; // Initialize output length
    if (j <= 0 || k <= 0 || j < k || j > MAX_ALLOWED_J) {
         fprintf(stderr, "Error: Invalid j (%d) or k (%d) parameters.\n", j, k);
        return NULL;
    }
    init_tables(); // Initialize nCk table etc. if not already done

    // Determine max number based on game type (simple check)
    int max_number = (strstr(game_type, "6_49")) ? 49 : 42; // Default to 6/42 if not 6/49
    if (max_number <= 0) {
         fprintf(stderr, "Error: Could not determine max_number for game_type '%s'.\n", game_type);
         return NULL;
    }


    if (draws_count < 1) {
        fprintf(stderr, "Error: No draws provided (draws_count = %d).\n", draws_count);
        return NULL;
    }

    // Validate last_offset
    if (last_offset < 0) last_offset = 0;
    if (last_offset >= draws_count) {
         fprintf(stderr, "Error: last_offset (%d) is too large for draws_count (%d).\n", last_offset, draws_count);
         // For standard analysis, this means no draws used. For chain, it's also invalid start.
         return NULL;
    }


    // Create a contiguous, sorted copy of the draw data needed for analysis.
    // The input `draws` is int**, which might not be contiguous.
    // We also sort numbers within each draw.
    int* sorted_draws_data = (int*)malloc(draws_count * 6 * sizeof(int));
    if (!sorted_draws_data) {
        perror("Failed to allocate memory for sorted draws");
        return NULL;
    }

    // Copy and sort each draw
    for (int i = 0; i < draws_count; i++) {
        int temp[6];
        if (draws[i] == NULL) {
             fprintf(stderr, "Error: NULL pointer for draw at index %d.\n", i);
             free(sorted_draws_data);
             return NULL;
        }
        // Copy numbers for sorting
        for (int z = 0; z < 6; z++) {
            temp[z] = draws[i][z];
            // Basic validation of numbers
            if (temp[z] < 1 || temp[z] > max_number) {
                 fprintf(stderr, "Warning: Invalid number %d in draw %d, position %d. Clamping/Ignoring might be needed.\n", temp[z], i, z);
                 // Decide handling: clamp, error out, or allow? Assuming DB ensures this.
                 // For now, let it pass, but pattern generation might warn/error later.
            }
        }

        // Simple bubble sort for the 6 numbers (or use qsort)
        for (int a = 0; a < 5; a++) {
            for (int b = a + 1; b < 6; b++) {
                if (temp[a] > temp[b]) {
                    int t = temp[a];
                    temp[a] = temp[b];
                    temp[b] = t;
                }
            }
        }
        // Copy sorted numbers into the contiguous array
        memcpy(&sorted_draws_data[i * 6], temp, 6 * sizeof(int));
    }

    AnalysisResultItem* result_ptr = NULL;

    // Dispatch to the correct analysis function based on 'l'
    if (l != -1) {
        // Standard Analysis (Top L + N)
        int analysis_use_count = draws_count - last_offset;
        if (analysis_use_count < 1) {
             fprintf(stderr, "Error: Standard analysis requires at least 1 draw after offset. use_count=%d\n", analysis_use_count);
             // No draws to analyze after applying offset
        } else {
            // We need to pass only the relevant part of sorted_draws_data
            // The standard analysis should use draws from index 'last_offset' to 'draws_count - 1'
             int* analysis_draws_start = &sorted_draws_data[0]; // Use draws from beginning up to use_count
            result_ptr = run_standard_analysis(
                analysis_draws_start, // Pass pointer to the start of relevant draws
                analysis_use_count,   // Number of draws to actually use
                j, k, m, l, n, max_number,
                out_len
            );
        }
    } else {
        // Chain Analysis (l = -1)
        // Chain analysis uses all draws but starts logic based on initial_offset
        result_ptr = run_chain_analysis(
            sorted_draws_data,      // Pass all sorted draws
            draws_count,            // Total count
            last_offset,            // Initial offset from the end
            j, k, m, max_number,
            out_len
        );
    }

    // Clean up the sorted draw data copy
    free(sorted_draws_data);

    return result_ptr; // Return the pointer to the results (or NULL)
}


// Function to free the memory allocated for analysis results
void free_analysis_results(AnalysisResultItem* results) {
    if (results) {
        free(results);
    }
}


// Helper to format a combination into a string "n1, n2, n3, ..."
static void format_combo(const int* combo, int len, char* out) {
    if (len <= 0) {
        out[0] = '\0';
        return;
    }
    int pos = 0;
    // Use snprintf for safety against buffer overflows
    for (int i = 0; i < len; i++) {
        int written = snprintf(out + pos, MAX_COMBO_STR - pos, "%s%d", (i > 0) ? ", " : "", combo[i]);
        if (written < 0 || written >= MAX_COMBO_STR - pos) {
             // Handle error or truncation
             out[MAX_COMBO_STR - 1] = '\0'; // Ensure null termination
             return;
        }
        pos += written;
    }
    // Ensure null termination if snprintf succeeded within bounds
    // out[pos] = '\0'; // Already handled by snprintf implicitly if space allows
}

// Helper to format the list of subsets and their ranks into a string
// "[( (s1,s2,..), rank1 ), ( (s1,s2,..), rank2 ), ...]" sorted by rank descending
static void format_subsets(const int* combo, int j, int k, int total_draws,
                          const SubsetTable* table, char* out) {
    typedef struct {
        int numbers[MAX_NUMBERS]; // Max k size needed
        int rank;
    } SubsetInfo;

    const int BUFFER_SIZE = MAX_SUBSETS_STR; // Max size for the output string

    if (j < k || k <= 0) { // Cannot form k-subsets
        strcpy(out, "[]");
        return;
    }

    uint64 exact_subset_count_64 = nCk_table[j][k];
    if (exact_subset_count_64 > INT_MAX || exact_subset_count_64 == 0) {
         fprintf(stderr,"Warning: Number of subsets C(%d,%d)=%llu is too large or zero.\n", j, k, exact_subset_count_64);
         strcpy(out, "[]"); // Or indicate error
         return;
    }
    int exact_subset_count = (int)exact_subset_count_64;


    SubsetInfo* subsets = (SubsetInfo*)malloc(exact_subset_count * sizeof(SubsetInfo));
    if (!subsets) {
        perror("Failed to allocate memory for subset formatting");
        strcpy(out, "[]"); // Indicate error
        return;
    }
    int subset_found_count = 0; // Track how many we actually process

    int indices[j]; // Indices to select k elements from the j numbers in combo
    for (int i = 0; i < k; i++) {
        indices[i] = i; // Start with the first k indices
    }

    // Iterate through all combinations of k indices from 0 to j-1
    while (subset_found_count < exact_subset_count) {
        uint64 subset_pattern = 0ULL;
        int current_subset_nums[k];
        int valid_subset = 1;

        // Build the subset pattern and store numbers
        for (int i = 0; i < k; i++) {
            int num = combo[indices[i]];
             if (num > 0 && num <= 64) {
                 subset_pattern |= (1ULL << (num - 1));
                 current_subset_nums[i] = num;
             } else {
                 fprintf(stderr,"Warning: Invalid number %d in combo during subset formatting.\n", num);
                 valid_subset = 0;
                 break;
             }
        }

        if (valid_subset) {
             // Lookup rank
             int last_seen = lookup_subset(table, subset_pattern);
             int rank = (last_seen >= 0) ? (total_draws - last_seen - 1) : total_draws;

             // Store subset numbers and rank
             memcpy(subsets[subset_found_count].numbers, current_subset_nums, k * sizeof(int));
             subsets[subset_found_count].rank = rank;
             subset_found_count++;
        }


        // Find the next combination of k indices
        int pos_to_increment = k - 1;
        while (pos_to_increment >= 0 && indices[pos_to_increment] == j - (k - pos_to_increment)) {
            pos_to_increment--;
        }
        if (pos_to_increment < 0) break; // No more combinations

        indices[pos_to_increment]++;
        for (int i = pos_to_increment + 1; i < k; i++) {
            indices[i] = indices[i - 1] + 1;
        }
    } // End while loop for combinations

    // Sort the found subsets by rank (descending) - simple bubble sort for illustration
    for (int i = 0; i < subset_found_count - 1; i++) {
        for (int l_idx = i + 1; l_idx < subset_found_count; l_idx++) { // Use l_idx instead of j
            if (subsets[l_idx].rank > subsets[i].rank) {
                SubsetInfo temp = subsets[i];
                subsets[i] = subsets[l_idx];
                subsets[l_idx] = temp;
            }
        }
    }

    // Format the sorted subsets into the output string
    int current_pos = 0;
    current_pos += snprintf(out + current_pos, BUFFER_SIZE - current_pos, "[");

    for (int i = 0; i < subset_found_count; i++) {
        if (current_pos >= BUFFER_SIZE - 1) break; // Stop if buffer nearly full

        if (i > 0) {
            current_pos += snprintf(out + current_pos, BUFFER_SIZE - current_pos, ", ");
             if (current_pos >= BUFFER_SIZE - 1) break;
        }

        current_pos += snprintf(out + current_pos, BUFFER_SIZE - current_pos, "((");
         if (current_pos >= BUFFER_SIZE - 1) break;

        // Format numbers in subset
        for (int n = 0; n < k; n++) {
            current_pos += snprintf(out + current_pos, BUFFER_SIZE - current_pos, "%s%d", (n > 0) ? ", " : "", subsets[i].numbers[n]);
            if (current_pos >= BUFFER_SIZE - 1) break;
        }
         if (current_pos >= BUFFER_SIZE - 1) break;


        current_pos += snprintf(out + current_pos, BUFFER_SIZE - current_pos, "), %d)", subsets[i].rank);
         if (current_pos >= BUFFER_SIZE - 1) break;

    }

    // Ensure closing bracket and null termination if space permits
    if (current_pos < BUFFER_SIZE - 1) {
        snprintf(out + current_pos, BUFFER_SIZE - current_pos, "]");
    } else {
         out[BUFFER_SIZE - 1] = '\0'; // Force termination if truncated
         out[BUFFER_SIZE - 2] = ']'; // Try to add bracket before termination
    }


    free(subsets); // Clean up allocated memory
}
