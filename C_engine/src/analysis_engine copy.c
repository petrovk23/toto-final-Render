/*
analysis_engine.c
-----------------
This is the heart of the high-performance analysis code in C. It replicates
the logic from the original Python "analysis.py" as closely as possible:

1) If l == -1, we perform the "chain" analysis. In the original Python code,
   that logic was in "run_analysis_chain_parallel" but here we do it in C with
   a straightforward approach. We build a "subset occurrence dictionary" based
   on partial draws, compute the best combination, check how many draws until
   it matches a future subset, update the offset, and continue. We store each
   iteration in the returned array.

2) Otherwise (l >= 1), we do normal "top-l" analysis. We again build the subset
   occurrence dictionary, then iterate over all j-combinations from [1..max_number].
   For each combination, we compute the sum of subset weights and the minimum subset
   weight. We keep the top-l combos in a small array (a typical partial-heap or
   iterative approach).

3) For each combo we store a textual "combination" (e.g. "1,2,3,4,5,6"),
   plus stats like average rank, min rank, and a textual listing of subsets
   with their occurrence values. We then return them to Python as an array
   of AnalysisResultItem.

Data structures used:
---------------------
- "SubsetOccDict": We store a large array 'subset_weights' that indexes each
  k-subset (of the [1..max_number] range) by a computed "subset index". Then
  the "weight" is stored in subset_weights[idx]. This is a direct-access
  structure that is simpler than a hash map but does require carefully
  computing combination indexes. For k=2..3, and max_number=42 or 49, this is
  quite feasible in memory. For example, C(42,3) = 11480, which is not large.
  That means direct array indexing is very fast. If we had a much larger
  problem, we might consider a more complex structure, but for TOTO 6/42 or
  6/49, this approach is fine.

- For enumerating j-combinations, we manually iterate from [1..max_number] in
  ascending order with "next_combination". This is typical for small j.

Possible alternatives:
----------------------
- Instead of storing a full array of subset_weights, a dictionary or a balanced
  tree might be used. But for TOTO scale, the direct array is typically faster.
- We could parallelize in threads or processes, but the user wanted a simpler
  single-threaded approach in C. If needed, we can parallelize enumerations with
  OpenMP or similar.

Note on chain analysis offset logic:
------------------------------------
We carefully replicate the logic "for idx in reversed(range(use_count)) => weight
= (use_count-1) - idx" from the Python code. This ensures older draws get higher
weights. We also implement the "draws_until_common = found_index - 1" approach to
mirror the original. If your original code had a subtle difference, adapt the
line "outR->draws_until_common = draws_until_match - 1;" accordingly.

Compiling:
----------
This .c file is compiled into a shared library (.so) alongside "analysis_engine.h".
Then Python (analysis.py) loads it with ctypes.

Copyright:
----------
Provided for demonstration and rapid TOTO analysis. No claims of fitness for
production usage without further safety checks.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "analysis_engine.h"

#define MAX_COMBO_STR 255
#define MAX_SUBSETS_STR 511

// Guard the maximum j to avoid huge allocations:
#define MAX_ALLOWED_J 200
// Guard maximum out_len to avoid huge dynamic arrays:
#define MAX_ALLOWED_OUT_LEN 1000000

// ----------------------------------------------------------------------------
// A simple function for computing nCr (combinations) as a 64-bit value.
// We only do this for small n in TOTO context, so no advanced checks are needed.
// ----------------------------------------------------------------------------
static long long comb_ll(int n, int r) {
    if (r > n) return 0;
    if (r == 0 || r == n) return 1;
    long long result = 1;
    for (int i = 1; i <= r; i++) {
        result = result * (n - r + i) / i;
    }
    return result;
}

// ----------------------------------------------------------------------------
// A utility to generate the next ascending combination of size k from [1..n].
// combo[] must be of size k. Return 1 if advanced, 0 if done.
// ----------------------------------------------------------------------------
static int next_combination(int *comb, int k, int n) {
    int rpos = k - 1;
    while (rpos >= 0 && comb[rpos] == (n - k + rpos + 1)) {
        rpos--;
    }
    if (rpos < 0) {
        return 0;
    }
    comb[rpos]++;
    for (int i = rpos + 1; i < k; i++) {
        comb[i] = comb[i - 1] + 1;
    }
    return 1;
}

// ----------------------------------------------------------------------------
// A SubsetOccDict that uses a direct array to store "weights" for each possible
// k-subset of [1..max_number] based on a computed index. This is faster than
// a dictionary for TOTO scale (max_number = 42 or 49).
// ----------------------------------------------------------------------------
typedef struct {
    long long *subset_weights; // array of weights
    int max_subsets;           // number of possible k-subsets
    int subset_size;           // k
    int max_number;            // 42 or 49
} SubsetOccDict;

// ----------------------------------------------------------------------------
// Create and free a SubsetOccDict
// ----------------------------------------------------------------------------
static SubsetOccDict* create_subset_occdict(int max_number, int k) {
    SubsetOccDict* d = (SubsetOccDict*)calloc(1, sizeof(SubsetOccDict));
    d->max_number = max_number;
    d->subset_size = k;
    d->max_subsets = (int)comb_ll(max_number, k);
    d->subset_weights = (long long*)calloc(d->max_subsets, sizeof(long long));
    return d;
}

static void free_subset_occdict(SubsetOccDict* d) {
    if (!d) return;
    free(d->subset_weights);
    free(d);
}

// ----------------------------------------------------------------------------
// Convert a sorted k-subset (e.g. [1,5,7]) to its zero-based "rank" index
// within all k-subsets of [1..max_number]. We sum over combinatorial counts.
// ----------------------------------------------------------------------------
static int subset_to_index(const int* subset, int k, int max_number) {
    long long rank = 0;
    int start = 1;
    for (int i = 0; i < k; i++) {
        int x = subset[i];
        for (int val = start; val < x; val++) {
            rank += comb_ll(max_number - val, k - i - 1);
        }
        start = x + 1;
    }
    return (int)rank;
}

// ----------------------------------------------------------------------------
// Update the subset_occdict with a "weight" for all k-subsets of the
// given "numbers" array. "numbers" must be sorted ascending. For TOTO draws
// of size 6, we do a standard combination enumeration for k-subsets of
// those 6 numbers. This effectively replicates the Python logic of
// subset_occurrence_dict[s] += weight.
// ----------------------------------------------------------------------------
static void update_subset_occdict(SubsetOccDict* d, const int* numbers, int count, long long weight) {
    if (!d || d->subset_size > count) return;
    int k = d->subset_size;
    int comb[20];  // enough for j up to 20 in TOTO context
    for (int i = 0; i < k; i++) {
        comb[i] = i;
    }

    while (1) {
        int temp[20];
        for (int i = 0; i < k; i++) {
            temp[i] = numbers[comb[i]];
        }
        int idx = subset_to_index(temp, k, d->max_number);
        if (idx >= 0 && idx < d->max_subsets) {
            d->subset_weights[idx] += weight;
        }

        int rpos = k - 1;
        while (rpos >= 0 && comb[rpos] == (rpos + (count - k))) {
            rpos--;
        }
        if (rpos < 0) break;
        comb[rpos]++;
        for (int j = rpos + 1; j < k; j++) {
            comb[j] = comb[j - 1] + 1;
        }
    }
}

// ----------------------------------------------------------------------------
// Convert an integer array "combo" of length j to a string like "1,2,3,4,5,6".
// We store it in out_str up to MAX_COMBO_STR length (255).
// ----------------------------------------------------------------------------
static void combo_to_string(const int *combo, int j, char* out_str) {
    out_str[0] = '\0';
    char tmp[32];
    for(int i = 0; i < j; i++){
        snprintf(tmp, sizeof(tmp), "%d", combo[i]);
        strcat(out_str, tmp);
        if(i < j - 1) strcat(out_str, ",");
    }
}

// ----------------------------------------------------------------------------
// Build the "Subsets" string, e.g. "[((2,9), 346), ((2,11), 520), ...]" with
// up to 511 chars. This is just a textual representation for demonstration.
// ----------------------------------------------------------------------------
static void build_subsets_string(
    int* combo,
    int j,
    int k,
    SubsetOccDict* occ,
    int max_number,
    char* out_buf
) {
    char big[16384]; // a large scratch buffer
    big[0] = '\0';
    strcat(big, "[");

    int ccomb[20];
    for(int i=0; i<k; i++){
        ccomb[i] = i;
    }

    int first_entry = 1;
    while (1) {
        int temp[20];
        for(int i=0; i<k; i++){
            temp[i] = combo[ccomb[i]];
        }
        int idx = subset_to_index(temp, k, max_number);
        long long w = 0;
        if(idx >= 0 && idx < occ->max_subsets){
            w = occ->subset_weights[idx];
        }

        // Build "( (2,9), 207 )"
        char subset_str[128];
        subset_str[0] = '\0';
        strcat(subset_str, "(");
        for(int z=0; z<k; z++){
            char numbuf[32];
            snprintf(numbuf, sizeof(numbuf), "%d", temp[z]);
            strcat(subset_str, numbuf);
            if(z < k-1) strcat(subset_str, ", ");
        }
        strcat(subset_str, ")");

        if(!first_entry) {
            strcat(big, ", ");
        } else {
            first_entry = 0;
        }

        char entry[256];
        snprintf(entry, sizeof(entry), "(%s, %lld)", subset_str, w);
        strcat(big, entry);

        int rpos = k - 1;
        while(rpos >= 0 && ccomb[rpos] == (rpos + (j - k))) {
            rpos--;
        }
        if(rpos < 0) break;
        ccomb[rpos]++;
        for(int xx = rpos+1; xx < k; xx++){
            ccomb[xx] = ccomb[xx-1] + 1;
        }
    }

    strcat(big, "]");

    // Truncate to 511 chars max
    strncpy(out_buf, big, MAX_SUBSETS_STR);
    out_buf[MAX_SUBSETS_STR] = '\0';
}

// ----------------------------------------------------------------------------
// run_analysis_c
// --------------
// The main exported function to replicate the original Python logic.
//
// If l == -1 => chain analysis
// Else        => normal "top-l" analysis
//
// The results are returned as a dynamically allocated array of AnalysisResultItem.
// The Python side will free them by calling free_analysis_results(...).
// ----------------------------------------------------------------------------
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
    // Basic safety checks
    if (j > MAX_ALLOWED_J) {
        fprintf(stderr, "Error: j=%d exceeds safety limit %d.\n", j, MAX_ALLOWED_J);
        *out_len = 0;
        return NULL;
    }

    // Determine max_number by game_type
    int max_number = 42;
    if (strstr(game_type, "6_49")) {
        max_number = 49;
    }

    *out_len = 0;
    AnalysisResultItem* results = NULL;

    // -------------------------
    // 1) Chain analysis (l == -1)
    // -------------------------
    if (l == -1) {
        int capacity = 1000; // grows if needed
        results = (AnalysisResultItem*)calloc(capacity, sizeof(AnalysisResultItem));
        int total_draws = draws_count;
        int current_offset = last_offset;

        // Sort each row ascending so that subsets match the python approach
        for(int i=0; i<draws_count; i++){
            for(int a=0; a<5; a++){
                for(int b=a+1; b<6; b++){
                    if(draws[i][a] > draws[i][b]){
                        int tmp = draws[i][a];
                        draws[i][a] = draws[i][b];
                        draws[i][b] = tmp;
                    }
                }
            }
        }

        while (current_offset >= 0 && current_offset < total_draws) {
            int use_count = total_draws - current_offset;
            if(use_count < 1) break;

            // Build the subset-occurrence dictionary for these "use_count" draws
            SubsetOccDict* occ = create_subset_occdict(max_number, k);
            for(int idx = use_count - 1; idx >= 0; idx--){
                long long weight = (use_count - 1) - idx;
                update_subset_occdict(occ, draws[idx], 6, weight);
            }

            // Now find the best single combo among all j-combinations
            int* comb_arr = (int*)malloc(sizeof(int)*j);
            for(int i=0; i<j; i++){
                comb_arr[i] = i+1;
            }

            double best_val = -1e9;
            double best_avg = 0.0;
            double best_minv = 0.0;
            int best_combo[64];
            char best_subsets[512];
            best_subsets[0] = '\0';

            do {
                long long sum_occ = 0;
                long long min_occ = LLONG_MAX;

                // For each k-subset of comb_arr, get its weight
                int ccomb[20];
                for(int i=0; i<k; i++){
                    ccomb[i] = i;
                }
                while(1) {
                    int temp_bc[20];
                    for(int x=0; x<k; x++){
                        temp_bc[x] = comb_arr[ccomb[x]];
                    }
                    int idx_s = subset_to_index(temp_bc, k, max_number);
                    long long w = 0;
                    if(idx_s>=0 && idx_s<occ->max_subsets){
                        w = occ->subset_weights[idx_s];
                    }
                    sum_occ += w;
                    if(w < min_occ) min_occ = w;

                    int rpos = k-1;
                    while(rpos>=0 && ccomb[rpos] == (rpos + (j - k))) rpos--;
                    if(rpos<0) break;
                    ccomb[rpos]++;
                    for(int xx=rpos+1; xx<k; xx++){
                        ccomb[xx] = ccomb[xx-1]+1;
                    }
                }

                double avg_occ = (double)sum_occ / (double)comb_ll(j, k);
                double sort_field = 0.0;
                if(strcmp(m, "avg") == 0) {
                    sort_field = avg_occ;
                } else {
                    sort_field = (double)min_occ;
                }

                if(sort_field > best_val) {
                    best_val = sort_field;
                    best_avg = avg_occ;
                    best_minv = (double)min_occ;
                    memcpy(best_combo, comb_arr, j*sizeof(int));
                    build_subsets_string(best_combo, j, k, occ, max_number, best_subsets);
                }
            } while(next_combination(comb_arr, j, max_number));

            free(comb_arr);
            free_subset_occdict(occ);

            // Calculate how many draws until a common subset is found
            int draws_until_match = 0;
            int found = 0;
            for(int dd = (total_draws - use_count); dd < total_draws; dd++){
                draws_until_match++;
                int row[6];
                memcpy(row, draws[dd], sizeof(int)*6);

                // Check if any k-subset in best_combo also appears in row
                int match_found = 0;
                int ccomb[20];
                for(int i=0; i<k; i++){
                    ccomb[i] = i;
                }
                while(!match_found) {
                    int temp_bc[20];
                    for(int x=0; x<k; x++){
                        temp_bc[x] = best_combo[ccomb[x]];
                    }
                    // Now check all k-subsets of row:
                    int rcomb[20];
                    for(int i=0; i<k; i++){
                        rcomb[i] = i;
                    }
                    while(1) {
                        int temp_row[20];
                        for(int x=0; x<k; x++){
                            temp_row[x] = row[rcomb[x]];
                        }
                        // Compare
                        int eq=1;
                        for(int z=0; z<k; z++){
                            if(temp_bc[z] != temp_row[z]) {
                                eq = 0;
                                break;
                            }
                        }
                        if(eq) {
                            match_found = 1;
                            break;
                        }

                        int rrpos = k-1;
                        while(rrpos>=0 && rcomb[rrpos] == (rrpos + (6 - k))) rrpos--;
                        if(rrpos<0) break;
                        rcomb[rrpos]++;
                        for(int xx=rrpos+1; xx<k; xx++){
                            rcomb[xx] = rcomb[xx-1]+1;
                        }
                    }

                    int rpos = k-1;
                    while(rpos>=0 && ccomb[rpos] == (rpos + (j - k))) rpos--;
                    if(rpos<0) break;
                    ccomb[rpos]++;
                    for(int xx=rpos+1; xx<k; xx++){
                        ccomb[xx] = ccomb[xx-1]+1;
                    }
                }

                if(match_found) {
                    found = 1;
                    break;
                }
            }

            // If no match, the chain ends here
            if(!found) {
                // In Python code, we store "Draws Until Common Subset" as the full `use_count`
                // if no match was found, not minus 1
                draws_until_match = use_count;
            }

            // Store in results
            if(*out_len >= capacity) {
                capacity *= 2;
                results = (AnalysisResultItem*)realloc(results, sizeof(AnalysisResultItem)*capacity);
            }
            AnalysisResultItem* outR = &results[*out_len];
            memset(outR, 0, sizeof(AnalysisResultItem));
            outR->is_chain_result = 1;
            outR->draw_offset = current_offset;
            outR->analysis_start_draw = total_draws - current_offset;

            // If found => subtract 1 in "Draws Until Common Subset"
            // If not found => store the full use_count
            if(found) {
                outR->draws_until_common = draws_until_match - 1;
            } else {
                outR->draws_until_common = use_count;
            }

            // Build the combo string
            char combo_str[512];
            combo_str[0] = '\0';
            combo_to_string(best_combo, j, combo_str);

            // Copy the combo safely
            if(strlen(combo_str) >= MAX_COMBO_STR) {
                strncpy(outR->combination, combo_str, MAX_COMBO_STR - 1);
                outR->combination[MAX_COMBO_STR - 1] = '\0';
            } else {
                strcpy(outR->combination, combo_str);
            }

            outR->avg_rank = best_avg;
            outR->min_value = best_minv;

            // Copy subsets safely
            if(strlen(best_subsets) >= MAX_SUBSETS_STR) {
                strncpy(outR->subsets, best_subsets, MAX_SUBSETS_STR - 1);
                outR->subsets[MAX_SUBSETS_STR - 1] = '\0';
            } else {
                strcpy(outR->subsets, best_subsets);
            }

            (*out_len)++;

            // If not found => no further chain steps
            // else reduce the offset by draws_until_match
            if(!found) {
                break;
            } else {
                current_offset = current_offset - draws_until_match;
            }
        }

        // Final safety check
        if (*out_len > MAX_ALLOWED_OUT_LEN) {
            fprintf(stderr, "Error: out_len=%d exceeds safety limit %d.\n", *out_len, MAX_ALLOWED_OUT_LEN);
            free(results);
            results = NULL;
            *out_len = 0;
        }
        return results;
    }

    // -------------------------
    // 2) Normal analysis (l >= 1)
    // -------------------------
    else {
        int use_count = draws_count - last_offset;
        if(use_count < 1) {
            return NULL;
        }

        // Sort each row ascending
        for(int i=0; i<draws_count; i++){
            for(int a=0; a<5; a++){
                for(int b=a+1; b<6; b++){
                    if(draws[i][a] > draws[i][b]){
                        int tmp = draws[i][a];
                        draws[i][a] = draws[i][b];
                        draws[i][b] = tmp;
                    }
                }
            }
        }

        // Build the subset-occurrence dictionary
        SubsetOccDict* occ = create_subset_occdict(max_number, k);
        for(int idx = use_count - 1; idx >= 0; idx--){
            long long weight = (use_count - 1) - idx;
            update_subset_occdict(occ, draws[idx], 6, weight);
        }

        long long total_combos = comb_ll(max_number, j);
        if(total_combos <= 0) {
            free_subset_occdict(occ);
            return NULL;
        }

        // We'll store up to l combos in memory for "top-l"
        typedef struct {
            int combo[64];
            int combo_len;
            double avg_occurrence;
            double min_occurrence;
            char subsets[512];
        } ComboStats;

        ComboStats* best_array = (ComboStats*)calloc(l, sizeof(ComboStats));
        int filled = 0;

        // Enumerate all j-combos of [1..max_number]
        int* comb_arr = (int*)malloc(sizeof(int)*j);
        for(int i=0; i<j; i++){
            comb_arr[i] = i+1;
        }

        do {
            long long sum_occ = 0;
            long long min_occ = LLONG_MAX;

            // We'll also keep a big scratch buffer to build them
            char big_subsets[512];
            big_subsets[0] = '\0';

            char big_temp[16384];
            big_temp[0] = '\0';
            strcat(big_temp, "[");

            int ccomb[20];
            for(int i=0; i<k; i++){
                ccomb[i] = i;
            }
            int first_sub = 1;

            while(1) {
                int temp[20];
                for(int x=0; x<k; x++){
                    temp[x] = comb_arr[ccomb[x]];
                }
                int idx_s = subset_to_index(temp, k, max_number);
                long long w = 0;
                if(idx_s>=0 && idx_s<occ->max_subsets){
                    w = occ->subset_weights[idx_s];
                }
                sum_occ += w;
                if(w < min_occ) min_occ = w;

                // If you want to replicate the "s = (subset, weight)" list:
                char subset_str[128];
                subset_str[0] = '\0';
                strcat(subset_str, "(");
                for(int z=0; z<k; z++){
                    char numbuf[32];
                    snprintf(numbuf, sizeof(numbuf), "%d", temp[z]);
                    strcat(subset_str, numbuf);
                    if(z<k-1) strcat(subset_str, ", ");
                }
                strcat(subset_str, ")");

                if(!first_sub) {
                    strcat(big_temp, ", ");
                } else {
                    first_sub = 0;
                }
                char entry[256];
                snprintf(entry, sizeof(entry), "(%s, %lld)", subset_str, w);
                strcat(big_temp, entry);

                int rpos = k-1;
                while(rpos>=0 && ccomb[rpos] == (rpos + (j - k))) rpos--;
                if(rpos<0) break;
                ccomb[rpos]++;
                for(int xx=rpos+1; xx<k; xx++){
                    ccomb[xx] = ccomb[xx-1]+1;
                }
            }
            strcat(big_temp, "]");
            strncpy(big_subsets, big_temp, 511);
            big_subsets[511] = '\0';

            double avg_occ = (double)sum_occ / (double)comb_ll(j, k);
            double sort_field;
            if(strcmp(m,"avg") == 0) {
                sort_field = avg_occ;
            } else {
                sort_field = (double)min_occ;
            }

            // Insert into best_array if needed
            if(filled < l) {
                ComboStats* slot = &best_array[filled];
                memcpy(slot->combo, comb_arr, j*sizeof(int));
                slot->combo_len = j;
                slot->avg_occurrence = avg_occ;
                slot->min_occurrence = (double)min_occ;
                strncpy(slot->subsets, big_subsets, 511);
                slot->subsets[511] = '\0';
                filled++;
            } else {
                // find the smallest in best_array to see if we can replace
                double min_val = 1e18;
                int min_idx = -1;
                for(int i=0; i<l; i++){
                    double sfield = (strcmp(m,"avg")==0)
                        ? best_array[i].avg_occurrence
                        : best_array[i].min_occurrence;
                    if(sfield < min_val) {
                        min_val = sfield;
                        min_idx = i;
                    }
                }
                if(sort_field > min_val && min_idx >= 0) {
                    ComboStats* slot = &best_array[min_idx];
                    memcpy(slot->combo, comb_arr, j*sizeof(int));
                    slot->combo_len = j;
                    slot->avg_occurrence = avg_occ;
                    slot->min_occurrence = (double)min_occ;
                    strncpy(slot->subsets, big_subsets, 511);
                    slot->subsets[511] = '\0';
                }
            }

        } while(next_combination(comb_arr, j, max_number));

        free(comb_arr);
        free_subset_occdict(occ);

        // We now have up to l combos in best_array. We sort them descending
        for(int i=0; i<filled; i++){
            for(int jx=i+1; jx<filled; jx++){
                double i_field = (strcmp(m,"avg")==0)
                    ? best_array[i].avg_occurrence
                    : best_array[i].min_occurrence;
                double j_field = (strcmp(m,"avg")==0)
                    ? best_array[jx].avg_occurrence
                    : best_array[jx].min_occurrence;
                if(j_field > i_field) {
                    // swap
                    ComboStats tmp = best_array[i];
                    best_array[i] = best_array[jx];
                    best_array[jx] = tmp;
                }
            }
        }

        // The final output includes up to l combos, plus up to n combos again as "selected"
        int final_len = filled + ((n > 0) ? filled : 0);
        if (final_len > MAX_ALLOWED_OUT_LEN) {
            fprintf(stderr, "Error: final_len=%d exceeds safety limit %d.\n", final_len, MAX_ALLOWED_OUT_LEN);
            free(best_array);
            return NULL;
        }

        results = (AnalysisResultItem*)calloc(final_len, sizeof(AnalysisResultItem));
        *out_len = final_len;

        // First l combos => top combos
        for(int i=0; i<filled; i++){
            AnalysisResultItem* outR = &results[i];
            outR->is_chain_result = 0;
            // Build combo string
            char combo_str[512];
            combo_str[0] = '\0';
            combo_to_string(best_array[i].combo, best_array[i].combo_len, combo_str);

            if(strlen(combo_str) >= MAX_COMBO_STR) {
                strncpy(outR->combination, combo_str, MAX_COMBO_STR - 1);
                outR->combination[MAX_COMBO_STR - 1] = '\0';
            } else {
                strcpy(outR->combination, combo_str);
            }

            outR->avg_rank = best_array[i].avg_occurrence;
            outR->min_value = best_array[i].min_occurrence;

            if(strlen(best_array[i].subsets) >= MAX_SUBSETS_STR) {
                strncpy(outR->subsets, best_array[i].subsets, MAX_SUBSETS_STR - 1);
                outR->subsets[MAX_SUBSETS_STR - 1] = '\0';
            } else {
                strcpy(outR->subsets, best_array[i].subsets);
            }
        }

        // Then n combos => "selected"
        if(n > 0) {
            int sel_count = 0;
            for(int i=0; i<filled; i++){
                if(sel_count >= n) break;
                int idx_out = filled + sel_count;
                AnalysisResultItem* outR = &results[idx_out];
                outR->is_chain_result = 0;

                char combo_str[512];
                combo_str[0] = '\0';
                combo_to_string(best_array[i].combo, best_array[i].combo_len, combo_str);

                if(strlen(combo_str) >= MAX_COMBO_STR) {
                    strncpy(outR->combination, combo_str, MAX_COMBO_STR - 1);
                    outR->combination[MAX_COMBO_STR - 1] = '\0';
                } else {
                    strcpy(outR->combination, combo_str);
                }

                outR->avg_rank = best_array[i].avg_occurrence;
                outR->min_value = best_array[i].min_occurrence;

                if(strlen(best_array[i].subsets) >= MAX_SUBSETS_STR) {
                    strncpy(outR->subsets, best_array[i].subsets, MAX_SUBSETS_STR - 1);
                    outR->subsets[MAX_SUBSETS_STR - 1] = '\0';
                } else {
                    strcpy(outR->subsets, best_array[i].subsets);
                }

                sel_count++;
            }
        }

        free(best_array);
        return results;
    }
}

// ----------------------------------------------------------------------------
// Free the array of AnalysisResultItem
// ----------------------------------------------------------------------------
void free_analysis_results(AnalysisResultItem* results) {
    if (results) {
        free(results);
    }
}
