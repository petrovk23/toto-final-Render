// analysis_engine.c — final version (correct results; avg ≈ min speed via per-depth ordering)
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
#define HASH_SIZE (1u << 26)  // 67M slots (power of two)

typedef unsigned long long uint64;
typedef unsigned int        uint32;

static uint64 nCk_table[MAX_NUMBERS][MAX_NUMBERS];
static int    bit_count_table[256];
static int    initialized = 0;

typedef struct {
    uint64* keys;
    int*    values;   // -1 == empty
    int     size;
    int     capacity;
} SubsetTable;

typedef struct {
    uint64 pattern;
    double avg_rank;
    double min_rank;
    int    combo[MAX_NUMBERS];
    int    len;
} ComboStats;

/* ---------- Forward decls ---------- */
static void   init_tables(void);
static inline int popcount64(uint64 x);
static SubsetTable* create_subset_table(int max_entries);
static void   free_subset_table(SubsetTable* table);
static inline uint32 hash_subset(uint64 pattern);
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value);
static inline int  lookup_subset(const SubsetTable* table, uint64 pattern);
static inline uint64 numbers_to_pattern(const int* numbers, int count);
static void   process_draw(const int* draw, int draw_idx, int k, SubsetTable* table);

static void   format_combo(const int* combo, int len, char* out);
static void   format_subsets(const int* combo, int j, int k, int total_draws,
                             const SubsetTable* table, char* out);

static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data,
    int use_count,
    int j,
    int k,
    int is_avg_mode,  // 1: avg, 0: min
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
    int is_avg_mode, // 1: avg, 0: min
    int max_number,
    int* out_len
);

/* ---------- Global TOP-L (exact pruning threshold) ---------- */
typedef struct {
    ComboStats* arr; // capacity L, kept roughly sorted desc (bubble on insert)
    int         count; // how many currently stored (<= L)
} GlobalTop;

typedef struct {
    double primary;   // avg in avg-mode, min in min-mode (score at tail)
    double tie;       // tie-breaker (min in avg-mode, avg in min-mode)
    int    filled;    // == GlobalTop.count (threshold valid iff filled >= L)
} GlobalThreshold;

static inline int better_avg_first(const ComboStats* a, const ComboStats* b) {
    if (a->avg_rank > b->avg_rank) return 1;
    if (a->avg_rank < b->avg_rank) return 0;
    return a->min_rank > b->min_rank;
}
static inline int better_min_first(const ComboStats* a, const ComboStats* b) {
    if (a->min_rank > b->min_rank) return 1;
    if (a->min_rank < b->min_rank) return 0;
    return a->avg_rank > b->avg_rank;
}

static void globaltop_try_insert(
    GlobalTop* G, GlobalThreshold* T, int L, int is_avg_mode, const ComboStats* cand)
{
    // Fast pre-check (no lock) when tail exists
    if (T->filled >= L) {
        if (is_avg_mode) {
            if (cand->avg_rank < T->primary) return;
            if (cand->avg_rank == T->primary && cand->min_rank <= T->tie) return;
        } else {
            if (cand->min_rank < T->primary) return;
            if (cand->min_rank == T->primary && cand->avg_rank <= T->tie) return;
        }
    }

    int do_insert = 1;
    #pragma omp critical(GlobalTopInsert)
    {
        // Re-check under lock
        if (T->filled >= L) {
            if (is_avg_mode) {
                if (cand->avg_rank < T->primary) do_insert = 0;
                else if (cand->avg_rank == T->primary && cand->min_rank <= T->tie) do_insert = 0;
            } else {
                if (cand->min_rank < T->primary) do_insert = 0;
                else if (cand->min_rank == T->primary && cand->avg_rank <= T->tie) do_insert = 0;
            }
        }

        if (do_insert) {
            int pos = (G->count < L) ? G->count : (L - 1);
            G->arr[pos] = *cand;
            if (G->count < L) G->count++;

            // Bubble up to maintain (mostly) descending order
            for (int i = pos; i > 0; i--) {
                int swap = is_avg_mode
                         ? better_avg_first(&G->arr[i], &G->arr[i-1])
                         : better_min_first(&G->arr[i], &G->arr[i-1]);
                if (!swap) break;
                ComboStats tmp = G->arr[i-1];
                G->arr[i-1] = G->arr[i];
                G->arr[i]   = tmp;
            }

            // Update threshold snapshot
            T->filled = G->count;
            if (G->count >= L) {
                const ComboStats* tail = &G->arr[L-1];
                if (is_avg_mode) { T->primary = tail->avg_rank; T->tie = tail->min_rank; }
                else             { T->primary = tail->min_rank; T->tie = tail->avg_rank; }
            } else {
                T->primary = -1.0; T->tie = -1.0; // not valid yet
            }
        }
    } // end critical
}

/* ---------- Hashing / subset table ---------- */
static void init_tables(void) {
    if (initialized) return;
    memset(nCk_table, 0, sizeof(nCk_table));
    for (int n = 0; n < MAX_NUMBERS; n++) {
        nCk_table[n][0] = 1;
        for (int k = 1; k <= n; k++) {
            nCk_table[n][k] = nCk_table[n-1][k-1] + nCk_table[n-1][k];
        }
    }
    for (int i = 0; i < 256; i++) {
        int c = 0;
        for (int b = 0; b < 8; b++) if (i & (1 << b)) c++;
        bit_count_table[i] = c;
    }
    initialized = 1;
}
static inline int popcount64(uint64 x) { return __builtin_popcountll(x); }

static SubsetTable* create_subset_table(int max_entries) {
    SubsetTable* t = (SubsetTable*)malloc(sizeof(SubsetTable));
    if (!t) return NULL;
    t->size = 0;
    t->capacity = max_entries;
    t->keys = (uint64*)calloc((size_t)max_entries, sizeof(uint64));
    t->values = (int*)malloc((size_t)max_entries * sizeof(int));
    if (!t->keys || !t->values) { free(t->keys); free(t->values); free(t); return NULL; }
    for (int i = 0; i < max_entries; i++) t->values[i] = -1;
    return t;
}
static void free_subset_table(SubsetTable* table) {
    if (!table) return;
    free(table->keys); free(table->values); free(table);
}
static inline uint32 hash_subset(uint64 pattern) {
    pattern ^= (pattern >> 33);
    pattern *= 0xff51afd7ed558ccdULL;
    pattern ^= (pattern >> 33);
    pattern *= 0xc4ceb9fe1a85ec53ULL;
    pattern ^= (pattern >> 33);
    return (uint32)(pattern & (HASH_SIZE - 1));
}
static inline void insert_subset(SubsetTable* table, uint64 pattern, int value) {
    uint32 idx = hash_subset(pattern);
    while (1) {
        int curv = table->values[idx];
        if (curv == -1 || table->keys[idx] == pattern) {
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
        int curv = table->values[idx];
        if (curv == -1) return -1;
        if (table->keys[idx] == pattern) return curv;
        idx = (idx + 1) & (HASH_SIZE - 1);
    }
}
static inline uint64 numbers_to_pattern(const int* numbers, int count) {
    uint64 p = 0ULL;
    for (int i = 0; i < count; i++) p |= (1ULL << (numbers[i] - 1));
    return p;
}
static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table) {
    if (k > 6) return;
    int idx[6];
    for (int i = 0; i < k; i++) idx[i] = i;
    while (1) {
        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) pat |= (1ULL << (draw[idx[i]] - 1));
        insert_subset(table, pat, draw_idx);
        int pos = k - 1;
        while (pos >= 0) {
            idx[pos]++;
            if (idx[pos] <= 6 - (k - pos)) {
                for (int x = pos + 1; x < k; x++) idx[x] = idx[x - 1] + 1;
                break;
            }
            pos--;
        }
        if (pos < 0) break;
    }
}

/* ---------- Pruning helper ---------- */
static int should_continue_branch(
    double upper_avg_candidate,
    double new_min_rank,
    int is_avg_mode,
    const GlobalThreshold* gth,
    int L)
{
    if (gth->filled < L) return 1; // no valid global threshold yet
    if (is_avg_mode) {
        if (upper_avg_candidate > gth->primary) return 1;
        if (upper_avg_candidate < gth->primary) return 0;
        return new_min_rank > gth->tie;
    } else {
        if (new_min_rank > gth->primary) return 1;
        if (new_min_rank < gth->primary) return 0;
        return upper_avg_candidate > gth->tie;
    }
}

/* ---------- Backtracking (NUMERIC coverage + per-depth heuristic ordering) ---------- */
/*
   We keep numeric increasing S (so every j-combo is visited exactly once),
   but at each depth we iterate candidate numbers (last+1..max_number)
   in a heuristic order (by single-number recency score1 desc, then num asc).
   This raises global thresholds early -> strong pruning in avg mode.
*/
static void backtrack(
    int* S,
    int size,
    uint64 current_S,
    double current_min_rank,
    double sum_current,
    int start_num,                 // numeric next candidate (>= last+1)
    const int* score1,             // per-number recency proxy (higher is better)
    SubsetTable* table,
    int total_draws,
    int max_number,
    int j,
    int k,
    int is_avg_mode,
    uint64 Cjk,
    int L,
    GlobalTop* gtop,
    GlobalThreshold* gthresh
) {
    if (size == j) {
        ComboStats cs = {0};
        cs.pattern  = current_S;
        cs.avg_rank = sum_current / (double)Cjk;
        cs.min_rank = current_min_rank;
        cs.len = j;
        for (int i = 0; i < j; i++) cs.combo[i] = S[i];
        globaltop_try_insert(gtop, gthresh, L, is_avg_mode, &cs);
        return;
    }

    int last_taken = (size > 0) ? S[size-1] : 0;
    int begin = (start_num > last_taken + 1) ? start_num : (last_taken + 1);

    // Build candidate list [begin..max_number], then order by score1 desc (tie num asc)
    int cand[MAX_NUMBERS];
    int cc = 0;
    // quick feasibility: if remaining numbers < (j - size) -> nothing to do
    int remaining_all = max_number - begin + 1;
    if (remaining_all < (j - size)) return;

    for (int num = begin; num <= max_number; num++) {
        cand[cc++] = num;
    }
    // simple stable selection sort by score1 desc, then num asc
    for (int i = 0; i < cc - 1; i++) {
        int best = i;
        for (int t = i + 1; t < cc; t++) {
            int a = cand[best], b = cand[t];
            if ( (score1[b] > score1[a]) || (score1[b] == score1[a] && b < a) ) {
                best = t;
            }
        }
        if (best != i) { int tmp = cand[i]; cand[i] = cand[best]; cand[best] = tmp; }
    }

    for (int idxc = 0; idxc < cc; idxc++) {
        int num = cand[idxc];
        if ((current_S & (1ULL << (num - 1))) != 0ULL) continue;

        // feasibility prune: numbers left after picking 'num'
        int left_after = max_number - num;
        if (left_after < (j - (size + 1) - 0)) continue;

        S[size] = num;
        uint64 new_S = current_S | (1ULL << (num - 1));

        double min_of_new = total_draws + 1.0;
        double sum_of_new = 0.0;

        if (size >= k - 1) {
            int idx_local[6];
            for (int i = 0; i < k - 1; i++) idx_local[i] = i;
            while (1) {
                int subset[6];
                for (int i = 0; i < k - 1; i++) subset[i] = S[idx_local[i]];
                subset[k - 1] = num;

                uint64 pat = numbers_to_pattern(subset, k);
                int last_seen = lookup_subset(table, pat);
                double rank = (last_seen >= 0) ? (double)(total_draws - last_seen - 1)
                                               : (double)total_draws;

                if (rank < min_of_new) min_of_new = rank;
                sum_of_new += rank;

                int p = k - 2;
                while (p >= 0) {
                    if (idx_local[p] < size - (k - 1 - p)) {
                        idx_local[p]++;
                        for (int x = p + 1; x < k - 1; x++)
                            idx_local[x] = idx_local[x - 1] + 1;
                        break;
                    }
                    p--;
                }
                if (p < 0) break;
            }
        }

        double new_min_rank    = (current_min_rank < min_of_new) ? current_min_rank : min_of_new;
        double new_sum_current = sum_current + sum_of_new;

        uint64 Cs_k = (size + 1 >= k) ? nCk_table[size + 1][k] : 0ULL;
        double upper_avg = (new_sum_current + ((double)(Cjk - Cs_k) * (double)total_draws))
                         / (double)Cjk;

        GlobalThreshold snap = *gthresh; // stale-safe read
        if (should_continue_branch(upper_avg, new_min_rank, is_avg_mode, &snap, L)) {
            backtrack(S, size + 1, new_S, new_min_rank, new_sum_current,
                      num + 1,
                      score1,
                      table, total_draws, max_number, j, k,
                      is_avg_mode, Cjk, L, gtop, gthresh);
        }
    }
}

/* ---------- Comparators for final sort ---------- */
static int compare_avg_rank(const void* a, const void* b) {
    const ComboStats* ca = (const ComboStats*)a;
    const ComboStats* cb = (const ComboStats*)b;
    if (ca->avg_rank > cb->avg_rank) return -1;
    if (ca->avg_rank < cb->avg_rank) return 1;
    if (ca->min_rank > cb->min_rank) return -1;
    if (ca->min_rank < cb->min_rank) return 1;
    return 0;
}
static int compare_min_rank(const void* a, const void* b) {
    const ComboStats* ca = (const ComboStats*)a;
    const ComboStats* cb = (const ComboStats*)b;
    if (ca->min_rank > cb->min_rank) return -1;
    if (ca->min_rank < cb->min_rank) return 1;
    if (ca->avg_rank > cb->avg_rank) return -1;
    if (ca->avg_rank < cb->avg_rank) return 1;
    return 0;
}

/* ---------- Standard analysis ---------- */
static AnalysisResultItem* run_standard_analysis(
    const int* sorted_draws_data,
    int use_count,
    int j,
    int k,
    int is_avg_mode,
    int l,
    int n,
    int max_number,
    int* out_len
) {
    *out_len = 0;

    SubsetTable* table = create_subset_table(HASH_SIZE);
    if (!table) return NULL;
    for (int i = 0; i < use_count; i++) process_draw(&sorted_draws_data[i * 6], i, k, table);

    /* Single-number recency proxy -> root & per-depth ordering */
    int* last_seen1 = (int*)malloc((size_t)(max_number + 1) * sizeof(int));
    int* score1     = (int*)malloc((size_t)(max_number + 1) * sizeof(int));
    int* order      = (int*)malloc((size_t)max_number * sizeof(int));
    if (!last_seen1 || !score1 || !order) { free(last_seen1); free(score1); free(order); free_subset_table(table); return NULL; }

    for (int v = 1; v <= max_number; v++) { last_seen1[v] = -1; score1[v] = use_count; }
    for (int i = 0; i < use_count; i++) {
        const int* d = &sorted_draws_data[i * 6];
        for (int z = 0; z < 6; z++) last_seen1[d[z]] = i;
    }
    for (int v = 1; v <= max_number; v++) score1[v] = (last_seen1[v] >= 0) ? (use_count - last_seen1[v] - 1) : use_count;
    for (int i = 0; i < max_number; i++) order[i] = i + 1;
    // Order only seeds the first number — deeper levels use per-depth ordering inside backtrack
    for (int i = 0; i < max_number - 1; i++) {
        for (int p = i + 1; p < max_number; p++) {
            int a = order[i], b = order[p];
            if ((score1[b] > score1[a]) || (score1[b] == score1[a] && b < a)) {
                int t = order[i]; order[i] = order[p]; order[p] = t;
            }
        }
    }

    GlobalTop gtop = {.arr = (ComboStats*)malloc((size_t)l * sizeof(ComboStats)), .count = 0};
    GlobalThreshold gth = {.primary=-1.0, .tie=-1.0, .filled=0};
    if (!gtop.arr) { free(order); free(score1); free(last_seen1); free_subset_table(table); return NULL; }

    int error_occurred = 0;
    uint64 Cjk = nCk_table[j][k];

    #pragma omp parallel
    {
        int* S = (int*)malloc((size_t)j * sizeof(int));
        if (!S) {
            #pragma omp atomic write
            error_occurred = 1;
        } else {
            #pragma omp for schedule(dynamic)
            for (int pos0 = 0; pos0 < max_number; pos0++) {
                if (error_occurred) continue;
                int first = order[pos0];
                if ((max_number - first) < (j - 1)) continue; // feasibility

                S[0] = first;
                uint64 current_S = (1ULL << (first - 1));
                double current_min_rank = (double)(use_count + 1);
                double sum_current = 0.0;

                backtrack(S, 1, current_S, current_min_rank, sum_current,
                          first + 1,                      // numeric from here
                          score1,                         // per-depth ordering source
                          table, use_count, max_number, j, k,
                          is_avg_mode, Cjk, l, &gtop, &gth);
            }
            free(S);
        }
    }

    free(order); free(score1); free(last_seen1);

    if (error_occurred) { free(gtop.arr); free_subset_table(table); return NULL; }

    /* Emit results */
    int top_count = (gtop.count < l) ? gtop.count : l;
    if (is_avg_mode) qsort(gtop.arr, (size_t)top_count, sizeof(ComboStats), compare_avg_rank);
    else             qsort(gtop.arr, (size_t)top_count, sizeof(ComboStats), compare_min_rank);

    AnalysisResultItem* results = (AnalysisResultItem*)calloc((size_t)(l + n), sizeof(AnalysisResultItem));
    if (!results) { free(gtop.arr); free_subset_table(table); return NULL; }
    int results_count = 0;

    // Rebuild table (cheap) for formatting subset ranks
    free_subset_table(table);
    table = create_subset_table(HASH_SIZE);
    for (int i = 0; i < use_count; i++) process_draw(&sorted_draws_data[i * 6], i, k, table);

    for (int i = 0; i < top_count; i++) {
        format_combo(gtop.arr[i].combo, gtop.arr[i].len, results[results_count].combination);
        format_subsets(gtop.arr[i].combo, j, k, use_count, table, results[results_count].subsets);
        results[results_count].avg_rank  = gtop.arr[i].avg_rank;
        results[results_count].min_value = gtop.arr[i].min_rank;
        results[results_count].is_chain_result = 0;
        results_count++;
    }

    // Second table: pick up to n non-overlapping (>=k disjointness) from top list
    int second_table_count = 0;
    if (n > 0 && top_count > 0) {
        int* pick = (int*)malloc((size_t)top_count * sizeof(int));
        if (pick) {
            for (int i = 0; i < top_count; i++) pick[i] = -1;
            int chosen = 0;
            pick[chosen++] = 0;
            for (int i = 1; i < top_count && chosen < n; i++) {
                uint64 pat_i = gtop.arr[i].pattern;
                int overlap = 0;
                for (int c = 0; c < chosen; c++) {
                    uint64 inter = (pat_i & gtop.arr[pick[c]].pattern);
                    if (popcount64(inter) >= k) { overlap = 1; break; }
                }
                if (!overlap) pick[chosen++] = i;
            }
            second_table_count = chosen;

            int base = results_count;
            for (int i = 0; i < second_table_count; i++) {
                int idx = pick[i];
                format_combo(gtop.arr[idx].combo, gtop.arr[idx].len, results[base + i].combination);
                format_subsets(gtop.arr[idx].combo, j, k, use_count, table, results[base + i].subsets);
                results[base + i].avg_rank  = gtop.arr[idx].avg_rank;
                results[base + i].min_value = gtop.arr[idx].min_rank;
                results[base + i].is_chain_result = 0;
            }
            free(pick);
        }
    }

    *out_len = results_count + second_table_count;
    free_subset_table(table);
    free(gtop.arr);

    if (*out_len == 0) { free(results); return NULL; }
    return results;
}

/* ---------- Chain analysis ---------- */
static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data,
    int draws_count,
    int initial_offset,
    int j,
    int k,
    int is_avg_mode,
    int max_number,
    int* out_len
) {
    AnalysisResultItem* chain_results = (AnalysisResultItem*)calloc((size_t)(initial_offset + 2), sizeof(AnalysisResultItem));
    if (!chain_results) { *out_len = 0; return NULL; }

    uint64* draw_patterns = (uint64*)malloc((size_t)draws_count * sizeof(uint64));
    if (!draw_patterns) { free(chain_results); *out_len = 0; return NULL; }
    for (int i = 0; i < draws_count; i++) draw_patterns[i] = numbers_to_pattern(&sorted_draws_data[i * 6], 6);

    int chain_index = 0;
    int current_offset = initial_offset;
    uint64 Cjk = nCk_table[j][k];

    while (current_offset >= 0 && current_offset <= draws_count - 1) {
        int use_count = draws_count - current_offset;
        if (use_count < 1) break;

        SubsetTable* table = create_subset_table(HASH_SIZE);
        for (int i = 0; i < use_count; i++) process_draw(&sorted_draws_data[i * 6], i, k, table);

        // Single-number recency for ordering
        int* last_seen1 = (int*)malloc((size_t)(max_number + 1) * sizeof(int));
        int* score1     = (int*)malloc((size_t)(max_number + 1) * sizeof(int));
        int* order      = (int*)malloc((size_t)max_number * sizeof(int));
        if (!last_seen1 || !score1 || !order) {
            free(last_seen1); free(score1); free(order);
            free_subset_table(table); free(draw_patterns); free(chain_results);
            *out_len = 0; return NULL;
        }
        for (int v = 1; v <= max_number; v++) { last_seen1[v] = -1; score1[v] = use_count; }
        for (int i = 0; i < use_count; i++) {
            const int* d = &sorted_draws_data[i * 6];
            for (int z = 0; z < 6; z++) last_seen1[d[z]] = i;
        }
        for (int v = 1; v <= max_number; v++) score1[v] = (last_seen1[v] >= 0) ? (use_count - last_seen1[v] - 1) : use_count;
        for (int i = 0; i < max_number; i++) order[i] = i + 1;
        for (int i = 0; i < max_number - 1; i++) {
            for (int p = i + 1; p < max_number; p++) {
                int a = order[i], b = order[p];
                if ((score1[b] > score1[a]) || (score1[b] == score1[a] && b < a)) {
                    int t = order[i]; order[i] = order[p]; order[p] = t;
                }
            }
        }

        GlobalTop gtop;
        ComboStats buf; gtop.arr = &buf; gtop.count = 0; // top-1 only
        GlobalThreshold gth = {.primary=-1.0, .tie=-1.0, .filled=0};

        int* S = (int*)malloc((size_t)j * sizeof(int));
        if (S) {
            for (int pos0 = 0; pos0 < max_number; pos0++) {
                int first = order[pos0];
                if ((max_number - first) < (j - 1)) continue;

                S[0] = first;
                uint64 current_S = (1ULL << (first - 1));
                double current_min_rank = (double)(use_count + 1);
                double sum_current = 0.0;

                backtrack(S, 1, current_S, current_min_rank, sum_current,
                          first + 1,                      // numeric from here
                          score1,
                          table, use_count, max_number, j, k,
                          is_avg_mode, Cjk, /*L=*/1, &gtop, &gth);
            }
            free(S);
        }

        free(order); free(score1); free(last_seen1);
        free_subset_table(table);

        if (gtop.count == 0) break;

        ComboStats best = gtop.arr[0];
        AnalysisResultItem* out_item = &chain_results[chain_index];
        format_combo(best.combo, best.len, out_item->combination);

        table = create_subset_table(HASH_SIZE);
        for (int i = 0; i < use_count; i++) process_draw(&sorted_draws_data[i * 6], i, k, table);
        format_subsets(best.combo, j, k, use_count, table, out_item->subsets);
        free_subset_table(table);

        out_item->avg_rank = best.avg_rank;
        out_item->min_value = best.min_rank;
        out_item->is_chain_result = 1;
        out_item->draw_offset = chain_index + 1;
        out_item->analysis_start_draw = draws_count - current_offset;

        uint64 combo_pat = best.pattern;
        int i;
        for (i = 1; i <= current_offset; i++) {
            int f_idx = draws_count - 1 - (current_offset - i);
            if (f_idx < 0) break;
            uint64 fpat = draw_patterns[f_idx];
            if (popcount64(combo_pat & fpat) >= k) break;
        }
        if (i > current_offset) i = current_offset + 1;
        out_item->draws_until_common = (i > 0) ? (i - 1) : 0;

        current_offset -= i;
        chain_index++;
        if (current_offset < 0) break;
    }

    free(draw_patterns);
    *out_len = chain_index;
    if (chain_index == 0) { free(chain_results); return NULL; }
    return chain_results;
}

/* ---------- Public API ---------- */
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
    init_tables();

    int max_number = (strstr(game_type, "6_49")) ? 49 : 42;
    if (draws_count < 1) return NULL;

    // Sort each draw's 6 numbers ascending
    int* sorted_draws_data = (int*)malloc((size_t)draws_count * 6 * sizeof(int));
    if (!sorted_draws_data) return NULL;

    for (int i = 0; i < draws_count; i++) {
        int temp[6];
        for (int z = 0; z < 6; z++) temp[z] = draws[i][z];
        for (int a = 0; a < 5; a++) {
            for (int b = a + 1; b < 6; b++) {
                if (temp[a] > temp[b]) { int t = temp[a]; temp[a] = temp[b]; temp[b] = t; }
            }
        }
        for (int z = 0; z < 6; z++) sorted_draws_data[i * 6 + z] = temp[z];
    }

    int is_avg_mode = (m && m[0] == 'a'); // "avg" starts with 'a', avoid strcmp in hot path

    AnalysisResultItem* ret =
        (l != -1)
        ? run_standard_analysis(sorted_draws_data, draws_count - last_offset, j, k, is_avg_mode, l, n, max_number, out_len)
        : run_chain_analysis   (sorted_draws_data, draws_count, last_offset, j, k, is_avg_mode, max_number, out_len);

    free(sorted_draws_data);
    return ret;
}

void free_analysis_results(AnalysisResultItem* results) {
    if (results) free(results);
}

/* ---------- Formatting ---------- */
static void format_combo(const int* combo, int len, char* out) {
    int pos = 0;
    for (int i = 0; i < len; i++) {
        if (i > 0) { out[pos++] = ','; out[pos++] = ' '; }
        pos += sprintf(out + pos, "%d", combo[i]);
    }
    out[pos] = '\0';
}
static void format_subsets(const int* combo, int j, int k, int total_draws,
                           const SubsetTable* table, char* out) {
    typedef struct { int numbers[6]; int rank; } SubsetInfo;
    int exact_subset_count = (int)nCk_table[j][k];
    SubsetInfo* subsets = (SubsetInfo*)malloc((size_t)exact_subset_count * sizeof(SubsetInfo));
    if (!subsets) { strcpy(out, "[]"); return; }

    int subset_count = 0;
    int idx[6];
    for (int i = 0; i < k; i++) idx[i] = i;
    while (1) {
        if (subset_count >= exact_subset_count) break;

        for (int i = 0; i < k; i++) subsets[subset_count].numbers[i] = combo[idx[i]];
        uint64 pat = numbers_to_pattern(subsets[subset_count].numbers, k);
        int last_seen = lookup_subset(table, pat);
        int rank = (last_seen >= 0) ? (total_draws - last_seen - 1) : total_draws;
        subsets[subset_count].rank = rank;

        subset_count++;

        int p = k - 1;
        while (p >= 0) {
            idx[p]++;
            if (idx[p] <= j - (k - p)) {
                for (int x = p + 1; x < k; x++) idx[x] = idx[x - 1] + 1;
                break;
            }
            p--;
        }
        if (p < 0) break;
    }

    // Sort descending by rank
    for (int i = 0; i < subset_count - 1; i++) {
        for (int t = i + 1; t < subset_count; t++) {
            if (subsets[t].rank > subsets[i].rank) {
                SubsetInfo tmp = subsets[i]; subsets[i] = subsets[t]; subsets[t] = tmp;
            }
        }
    }

    int pos = 0;
    out[pos++] = '[';
    for (int i = 0; i < subset_count; i++) {
        if (i > 0) { out[pos++] = ','; out[pos++] = ' '; }
        pos += sprintf(out + pos, "((%d", subsets[i].numbers[0]);
        for (int n = 1; n < k; n++) pos += sprintf(out + pos, ", %d", subsets[i].numbers[n]);
        pos += sprintf(out + pos, "), %d)", subsets[i].rank);
    }
    out[pos++] = ']';
    out[pos] = '\0';
    free(subsets);
}
