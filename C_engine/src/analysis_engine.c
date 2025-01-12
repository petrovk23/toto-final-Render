#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "analysis_engine.h"

/*
 * This version replicates your Python logic exactly (for normal and chain analysis),
 * but also:
 *   - Manually ensures null termination in strncpy calls (no truncation warning).
 *   - Adds simple sanity checks for j and out_len to silence large allocation warnings.
 * Thus, it returns the same results as your second screenshot while avoiding warnings.
 */

#define MAX_COMBO_STR 255   // combination[] field length
#define MAX_SUBSETS_STR 511 // subsets[] field length

// If you want a different upper bound, adjust these:
#define MAX_ALLOWED_J 200      // combos bigger than 200 are unrealistic for TOTO
#define MAX_ALLOWED_OUT_LEN 1000000  // 1 million, just a safe upper bound

/*
 * Compute nCr (combinations).
 */
static long long comb_ll(int n, int r) {
    if (r > n) return 0;
    if (r == 0 || r == n) return 1;
    long long result = 1;
    for (int i = 1; i <= r; i++) {
        result = result * (n - r + i) / i;
    }
    return result;
}

/*
 * Move from one ascending combination to the next.
 * Returns 1 if advanced, 0 if done.
 */
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

/*
 * SubsetOccDict for storing k-subset weights.
 */
typedef struct {
    long long *subset_weights;
    int max_subsets;
    int subset_size;
    int max_number;
} SubsetOccDict;

/*
 * Create / free.
 */
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

/*
 * Map a sorted subset[] of size k to an index in [0..comb(max_number,k)-1].
 */
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

/*
 * Add 'weight' to each k-subset of 'numbers'.
 * 'numbers' is sorted, up to 6 elements for TOTO draws.
 */
static void update_subset_occdict(SubsetOccDict* d, const int* numbers, int count, long long weight) {
    if (!d || d->subset_size > count) return;

    int k = d->subset_size;
    int comb[20];
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

/*
 * Convert an integer combo to "1,2,3,4,5,6".
 */
static void combo_to_string(const int *combo, int j, char* out_str) {
    out_str[0] = '\0';
    char tmp[32];
    for(int i = 0; i < j; i++){
        snprintf(tmp, sizeof(tmp), "%d", combo[i]);
        strcat(out_str, tmp);
        if(i < j-1) strcat(out_str, ",");
    }
}

/*
 * Builds something like "[((2,9,14), 346), ((2,9,21), 281), ...]" for the subsets.
 */
static void build_subsets_string(
    int* combo,
    int j,
    int k,
    SubsetOccDict* occ,
    int max_number,
    char* out_buf
) {
    char big[16384];
    big[0] = '\0';
    strcat(big, "[");

    int ccomb[20];
    for(int i=0; i<k; i++){
        ccomb[i] = i;
    }

    int first_entry = 1;
    while (1) {
        int temp[20];
        for(int x=0; x<k; x++){
            temp[x] = combo[ccomb[x]];
        }
        int idx = subset_to_index(temp, k, max_number);
        long long w = 0;
        if(idx >= 0 && idx < occ->max_subsets){
            w = occ->subset_weights[idx];
        }

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

    // Truncate to 511 chars, ensure null termination
    strncpy(out_buf, big, MAX_SUBSETS_STR);
    out_buf[MAX_SUBSETS_STR] = '\0';
}

/*
 * The main analysis function. Exactly replicates your original Python weighting logic:
 * Chain (l == -1) or Normal (top-l).
 */
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
    // Quick checks to silence large-allocation warnings
    if (j > MAX_ALLOWED_J) {
        fprintf(stderr, "Error: j=%d exceeds safety limit %d.\n", j, MAX_ALLOWED_J);
        *out_len = 0;
        return NULL;
    }

    int max_number = 42;
    if (strstr(game_type, "6_49")) {
        max_number = 49;
    }

    *out_len = 0;
    AnalysisResultItem* results = NULL;

    // CHAIN analysis if l == -1
    if (l == -1) {
        int capacity = 1000;
        results = (AnalysisResultItem*)calloc(capacity, sizeof(AnalysisResultItem));
        int total_draws = draws_count;
        int current_offset = last_offset;

        // Sort ascending
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

            SubsetOccDict* occ = create_subset_occdict(max_number, k);
            // Weighted approach: for idx in reversed(range(use_count))
            for(int idx = use_count - 1; idx >= 0; idx--){
                long long weight = (use_count - 1) - idx;
                update_subset_occdict(occ, draws[idx], 6, weight);
            }

            // Find best combo
            int* comb_arr = (int*)malloc(sizeof(int)*j);
            for(int i=0; i<j; i++){
                comb_arr[i] = i+1;
            }

            double best_val = -1e9;
            double best_avg = 0.0;
            double best_minv = 0.0;
            int best_combo[64];
            char best_subsets[512] = {0};

            do {
                long long sum_occ = 0;
                long long min_occ = LLONG_MAX;

                int ccomb[20];
                for(int i=0; i<k; i++){
                    ccomb[i] = i;
                }
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

            // draws_until_match
            int draws_until_match = 0;
            int found = 0;
            for(int dd = (total_draws - use_count); dd < total_draws; dd++){
                draws_until_match++;
                int row[6];
                memcpy(row, draws[dd], sizeof(int)*6);

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
                    int rcomb[20];
                    for(int i=0; i<k; i++){
                        rcomb[i] = i;
                    }
                    while(1) {
                        int temp_row[20];
                        for(int x=0; x<k; x++){
                            temp_row[x] = row[rcomb[x]];
                        }
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
            if(!found) {
                draws_until_match = use_count;
            }

            // Add chain result
            if(*out_len >= capacity) {
                capacity *= 2;
                results = (AnalysisResultItem*)realloc(results, sizeof(AnalysisResultItem)*capacity);
            }
            AnalysisResultItem* outR = &results[*out_len];
            memset(outR, 0, sizeof(AnalysisResultItem));
            outR->is_chain_result = 1;
            outR->draw_offset = current_offset;
            outR->analysis_start_draw = total_draws - current_offset;
            outR->draws_until_common = draws_until_match - 1;

            char combo_str[512] = {0};
            combo_to_string(best_combo, j, combo_str);
            // Copy safely
            if(strlen(combo_str) >= MAX_COMBO_STR) {
                strncpy(outR->combination, combo_str, MAX_COMBO_STR - 1);
                outR->combination[MAX_COMBO_STR - 1] = '\0';
            } else {
                strcpy(outR->combination, combo_str);
            }

            outR->avg_rank = best_avg;
            outR->min_value = best_minv;

            if(strlen(best_subsets) >= MAX_SUBSETS_STR) {
                strncpy(outR->subsets, best_subsets, MAX_SUBSETS_STR - 1);
                outR->subsets[MAX_SUBSETS_STR - 1] = '\0';
            } else {
                strcpy(outR->subsets, best_subsets);
            }

            (*out_len)++;

            if(!found) {
                break;
            } else {
                current_offset = current_offset - draws_until_match;
            }
        }

        // Final check for out_len
        if (*out_len > MAX_ALLOWED_OUT_LEN) {
            fprintf(stderr, "Error: out_len=%d exceeds safety limit %d.\n", *out_len, MAX_ALLOWED_OUT_LEN);
            free(results);
            results = NULL;
            *out_len = 0;
        }
        return results;
    }
    // NORMAL analysis
    else {
        int use_count = draws_count - last_offset;
        if(use_count < 1) {
            return NULL;
        }
        // sort ascending
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

        SubsetOccDict* occ = create_subset_occdict(max_number, k);

        // EXACT Python weighting: reversed
        for(int idx = use_count - 1; idx >= 0; idx--){
            long long weight = (use_count - 1) - idx;
            update_subset_occdict(occ, draws[idx], 6, weight);
        }

        long long total_combos = comb_ll(max_number, j);
        if(total_combos <= 0) {
            free_subset_occdict(occ);
            return NULL;
        }

        // We'll store up to l combos
        typedef struct {
            int combo[64];
            int combo_len;
            double avg_occurrence;
            double min_occurrence;
            char subsets[512];
        } ComboStats;

        ComboStats* best_array = (ComboStats*)calloc(l, sizeof(ComboStats));
        int filled = 0;

        int* comb_arr = (int*)malloc(sizeof(int)*j);
        for(int i=0; i<j; i++){
            comb_arr[i] = i+1;
        }

        do {
            long long sum_occ = 0;
            long long min_occ = LLONG_MAX;

            // Build subsets
            char big_subsets[512] = {0};
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

                // add to big_temp
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

                int rpos = k - 1;
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
            double sort_field = 0.0;
            if(strcmp(m,"avg") == 0) {
                sort_field = avg_occ;
            } else {
                sort_field = (double)min_occ;
            }

            if(filled < l) {
                ComboStats* slot = &best_array[filled];
                memcpy(slot->combo, comb_arr, j*sizeof(int));
                slot->combo_len = j;
                slot->avg_occurrence = avg_occ;
                slot->min_occurrence = min_occ;
                strncpy(slot->subsets, big_subsets, 511);
                slot->subsets[511] = '\0';
                filled++;
            } else {
                // find smallest
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
                if(sort_field > min_val && min_idx>=0) {
                    ComboStats* slot = &best_array[min_idx];
                    memcpy(slot->combo, comb_arr, j*sizeof(int));
                    slot->combo_len = j;
                    slot->avg_occurrence = avg_occ;
                    slot->min_occurrence = min_occ;
                    strncpy(slot->subsets, big_subsets, 511);
                    slot->subsets[511] = '\0';
                }
            }

        } while(next_combination(comb_arr, j, max_number));

        free(comb_arr);
        free_subset_occdict(occ);

        // Sort best_array descending
        for(int i=0; i<filled; i++){
            for(int jx=i+1; jx<filled; jx++){
                double i_field = (strcmp(m,"avg")==0)
                                 ? best_array[i].avg_occurrence
                                 : best_array[i].min_occurrence;
                double j_field = (strcmp(m,"avg")==0)
                                 ? best_array[jx].avg_occurrence
                                 : best_array[jx].min_occurrence;
                if(j_field > i_field) {
                    ComboStats tmp = best_array[i];
                    best_array[i] = best_array[jx];
                    best_array[jx] = tmp;
                }
            }
        }

        // We produce up to l combos plus n combos
        int final_len = filled + ((n > 0) ? filled : 0);

        if (final_len > MAX_ALLOWED_OUT_LEN) {
            fprintf(stderr, "Error: final_len=%d exceeds safety limit %d.\n", final_len, MAX_ALLOWED_OUT_LEN);
            free(best_array);
            return NULL;
        }

        results = (AnalysisResultItem*)calloc(final_len, sizeof(AnalysisResultItem));
        *out_len = final_len;

        // top combos
        for(int i=0; i<filled; i++){
            AnalysisResultItem* outR = &results[i];
            outR->is_chain_result = 0;

            // Build combo string
            char combo_str[512] = {0};
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

        // n "selected combos" (simple approach)
        if(n > 0) {
            int sel_count = 0;
            for(int i=0; i<filled; i++){
                if(sel_count >= n) break;
                int idx_out = filled + sel_count;
                AnalysisResultItem* outR = &results[idx_out];
                outR->is_chain_result = 0;

                char combo_str[512] = {0};
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

/*
 * Free the array returned by run_analysis_c().
 */
void free_analysis_results(AnalysisResultItem* results) {
    if (results) free(results);
}
