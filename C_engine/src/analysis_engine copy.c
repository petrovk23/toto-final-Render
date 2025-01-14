#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "analysis_engine.h"

#define MAX_NUMBERS 50
#define MAX_SUBSET_STR 512
#define HASH_SIZE (1 << 20)  // 1M entries
#define CACHE_LINE 64
#define VECTOR_SIZE 256

// Cache-aligned subset table entry
typedef struct __attribute__((aligned(32))) {
    __m256i pattern;  // 256-bit pattern for AVX2 operations
    uint32_t last_seen;
} SubsetEntry;

// Cache-aligned subset table
typedef struct __attribute__((aligned(32))) {
    SubsetEntry* entries;
    size_t size;
    size_t capacity;
} SubsetTable;

// Cache-aligned combo statistics for vectorized operations
typedef struct __attribute__((aligned(32))) {
    __m256i pattern;
    double avg_rank;
    double min_rank;
    int combo[8];  // padded for alignment
    int length;
} ComboStats;

// Pre-computed tables
static uint64_t choose_table[MAX_NUMBERS][MAX_NUMBERS];
static uint8_t popcount_table[256];
static int initialized = 0;

// Initialize lookup tables
static void init_tables(void) {
    if (initialized) return;

    // Initialize combinatorial numbers
    for (int n = 0; n < MAX_NUMBERS; n++) {
        choose_table[n][0] = 1;
        choose_table[n][n] = 1;
        for (int k = 1; k < n; k++) {
            choose_table[n][k] = choose_table[n-1][k-1] + choose_table[n-1][k];
        }
    }

    // Initialize popcount lookup
    for (int i = 0; i < 256; i++) {
        int count = 0;
        for (int j = 0; j < 8; j++) {
            if (i & (1 << j)) count++;
        }
        popcount_table[i] = count;
    }

    initialized = 1;
}

// SIMD-optimized popcount
static inline int popcount256(__m256i v) {
    __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );

    __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);

    __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i sum = _mm256_add_epi8(popcnt1, popcnt2);

    // Horizontal sum of bytes
    __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(sum),
        _mm256_extracti128_si256(sum, 1)
    );
    return _mm_cvtsi128_si32(sum128);
}

// Convert numbers to bit pattern optimized for SIMD
static inline __m256i numbers_to_pattern_simd(const int* nums, int count) {
    uint32_t pattern[8] = {0};  // AVX2 register size
    for (int i = 0; i < count; i++) {
        pattern[nums[i] / 32] |= 1U << (nums[i] % 32);
    }
    return _mm256_loadu_si256((__m256i*)pattern);
}

// Create subset table
static SubsetTable* create_subset_table(size_t capacity) {
    SubsetTable* table = (SubsetTable*)_mm_malloc(sizeof(SubsetTable), 32);
    if (!table) return NULL;

    table->entries = (SubsetEntry*)_mm_malloc(capacity * sizeof(SubsetEntry), 32);
    if (!table->entries) {
        _mm_free(table);
        return NULL;
    }

    table->capacity = capacity;
    table->size = 0;

    memset(table->entries, 0, capacity * sizeof(SubsetEntry));
    return table;
}

// Free subset table
static void free_subset_table(SubsetTable* table) {
    if (table) {
        if (table->entries) _mm_free(table->entries);
        _mm_free(table);
    }
}

// SIMD-optimized subset lookup
static inline uint32_t lookup_subset(const SubsetTable* table, __m256i pattern) {
    uint32_t hash = _mm256_extract_epi32(_mm256_xor_si256(pattern,
        _mm256_srli_epi32(pattern, 13)), 0) & (table->capacity - 1);

    while (1) {
        __m256i stored = table->entries[hash].pattern;
        if (_mm256_testc_si256(pattern, stored)) {
            return table->entries[hash].last_seen;
        }
        if (table->entries[hash].last_seen == UINT32_MAX) {
            return UINT32_MAX;
        }
        hash = (hash + 1) & (table->capacity - 1);
    }
}

// Process draw and update subset table
static void process_draw(const int* draw, int draw_idx, int k,
                        SubsetTable* table) {
    int indices[6] = {0, 1, 2, 3, 4, 5};
    int pos = k - 1;

    do {
        __m256i pattern = numbers_to_pattern_simd(draw, k);

        uint32_t hash = _mm256_extract_epi32(_mm256_xor_si256(pattern,
            _mm256_srli_epi32(pattern, 13)), 0) & (table->capacity - 1);

        while (1) {
            if (table->entries[hash].last_seen == UINT32_MAX ||
                _mm256_testc_si256(pattern, table->entries[hash].pattern)) {
                table->entries[hash].pattern = pattern;
                table->entries[hash].last_seen = draw_idx;
                break;
            }
            hash = (hash + 1) & (table->capacity - 1);
        }

        // Next combination
        while (pos >= 0 && indices[pos] >= 6 - k + pos) pos--;
        if (pos < 0) break;

        indices[pos]++;
        for (int i = pos + 1; i < k; i++) {
            indices[i] = indices[i-1] + 1;
        }
        pos = k - 1;
    } while (1);
}

// SIMD-optimized combo evaluation
static void evaluate_combo(const int* combo, int j, int k, int draws_count,
                         const SubsetTable* table, double* avg_rank,
                         double* min_rank) {
    __m256d sum = _mm256_setzero_pd();
    __m256d minv = _mm256_set1_pd(draws_count);
    int count = 0;

    int indices[8] = {0};  // k is typically small
    for (int i = 0; i < k; i++) indices[i] = i;

    do {
        __m256i pattern = numbers_to_pattern_simd(combo, k);
        uint32_t last_seen = lookup_subset(table, pattern);

        double rank = (last_seen != UINT32_MAX) ?
                     (double)(draws_count - last_seen - 1) : draws_count;

        sum = _mm256_add_pd(sum, _mm256_set1_pd(rank));
        minv = _mm256_min_pd(minv, _mm256_set1_pd(rank));
        count++;

        // Next k-subset
        int i = k - 1;
        while (i >= 0 && indices[i] == j - k + i) i--;
        if (i < 0) break;

        indices[i]++;
        for (int x = i + 1; x < k; x++) {
            indices[x] = indices[i] + (x - i);
        }
    } while (1);

    double sum_arr[4], min_arr[4];
    _mm256_store_pd(sum_arr, sum);
    _mm256_store_pd(min_arr, minv);

    *avg_rank = sum_arr[0] / count;
    *min_rank = min_arr[0];
}

// Process combinations in parallel with SIMD
static void process_combinations(int j, int k, int max_number,
                               const SubsetTable* table, int draws_count,
                               const char* mode, AnalysisResultItem* results,
                               int limit, int* out_count) {
    const int block_size = 1024;
    *out_count = 0;

    #pragma omp parallel
    {
        ComboStats* local_best = (ComboStats*)_mm_malloc(
            limit * sizeof(ComboStats), 32);
        int local_count = 0;

        #pragma omp for schedule(dynamic, block_size)
        for (int first = 1; first <= max_number - j + 1; first++) {
            int combo[MAX_NUMBERS];
            combo[0] = first;

            void generate_rest(int pos) {
                if (pos == j) {
                    double avg_rank, min_rank;
                    evaluate_combo(combo, j, k, draws_count, table,
                                 &avg_rank, &min_rank);

                    if (local_count < limit) {
                        ComboStats* stat = &local_best[local_count++];
                        memcpy(stat->combo, combo, j * sizeof(int));
                        stat->pattern = numbers_to_pattern_simd(combo, j);
                        stat->avg_rank = avg_rank;
                        stat->min_rank = min_rank;
                        stat->length = j;
                    } else {
                        // Replace worst if better
                        int worst_idx = 0;
                        double worst_val = strcmp(mode, "avg") == 0 ?
                            local_best[0].avg_rank : local_best[0].min_rank;

                        for (int i = 1; i < limit; i++) {
                            double val = strcmp(mode, "avg") == 0 ?
                                local_best[i].avg_rank : local_best[i].min_rank;
                            if (val < worst_val) {
                                worst_val = val;
                                worst_idx = i;
                            }
                        }

                        double compare_val = strcmp(mode, "avg") == 0 ?
                            avg_rank : min_rank;

                        if (compare_val > worst_val) {
                            ComboStats* stat = &local_best[worst_idx];
                            memcpy(stat->combo, combo, j * sizeof(int));
                            stat->pattern = numbers_to_pattern_simd(combo, j);
                            stat->avg_rank = avg_rank;
                            stat->min_rank = min_rank;
                            stat->length = j;
                        }
                    }
                    return;
                }

                for (int i = combo[pos-1] + 1; i <= max_number - (j-pos); i++) {
                    combo[pos] = i;
                    generate_rest(pos + 1);
                }
            }

            generate_rest(1);
        }

        // Merge results
        #pragma omp critical
        {
            for (int i = 0; i < local_count && *out_count < limit; i++) {
                ComboStats* stat = &local_best[i];
                AnalysisResultItem* item = &results[*out_count];

                char* ptr = item->combination;
                for (int x = 0; x < stat->length; x++) {
                    if (x > 0) *ptr++ = ',';
                    ptr += sprintf(ptr, "%d", stat->combo[x]);
                }
                *ptr = '\0';

                item->avg_rank = stat->avg_rank;
                item->min_value = stat->min_rank;
                item->is_chain_result = 0;
                (*out_count)++;
            }
        }

        _mm_free(local_best);
    }
}

// Main entry point from Python
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
    init_tables();

    int max_number = strstr(game_type, "6_49") ? 49 : 42;

    if (last_offset < 0) last_offset = 0;
    if (last_offset > draws_count) last_offset = draws_count;

    int use_count = draws_count - last_offset;
    if (use_count < 1) return NULL;

    // Create subset table
    SubsetTable* table = create_subset_table(HASH_SIZE);
    if (!table) return NULL;

    // Process draws
    for (int i = 0; i < use_count; i++) {
        process_draw(draws[i], i, k, table);
    }

    // Allocate results
    int capacity = (l == -1) ? draws_count : (l + n);
    AnalysisResultItem* results = (AnalysisResultItem*)_mm_malloc(
        capacity * sizeof(AnalysisResultItem), 32);

    if (!results) {
        free_subset_table(table);
        return NULL;
    }

    memset(results, 0, capacity * sizeof(AnalysisResultItem));

    if (l == -1) {
        // Chain analysis
        int chain_count = 0;
        int current_offset = last_offset;

        while (current_offset < draws_count && chain_count < capacity) {
            // Find best combo at current offset
            int local_count = 0;
            process_combinations(j, k, max_number, table,
                              draws_count - current_offset,
                              m, &results[chain_count], 1, &local_count);

            if (local_count == 0) break;

            results[chain_count].is_chain_result = 1;
            results[chain_count].draw_offset = current_offset;
            results[chain_count].analysis_start_draw = draws_count - current_offset;

            // Find next match
            int next_combo[MAX_NUMBERS];
            const char* p = results[chain_count].combination;
            int num_count = 0;

            while (*p && num_count < j) {
                next_combo[num_count++] = atoi(p);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }

            int match_found = 0;
            int match_draws = draws_count - current_offset;

            for (int i = 1; i < draws_count - current_offset && !match_found; i++) {
                __m256i draw_pattern = numbers_to_pattern_simd(draws[i], 6);

                int indices[MAX_NUMBERS];
                for (int x1 = 0; x1 < k; x1++) indices[x1] = x1;

                do {
                    __m256i subset = numbers_to_pattern_simd(&next_combo[indices[0]], k);

                    if (_mm256_testc_si256(subset, draw_pattern)) {
                        match_found = 1;
                        match_draws = i;
                        break;
                    }

                    // Next k-combination
                    int p = k - 1;
                    while (p >= 0 && indices[p] == j - k + p) p--;
                    if (p < 0) break;
                    indices[p]++;
                    for (int x = p + 1; x < k; x++) {
                        indices[x] = indices[p] + (x - p);
                    }
                } while (1);

                if (match_found) break;
            }

            results[chain_count].draws_until_common = match_draws;

            if (!match_found) {
                chain_count++;
                break;
            }

            current_offset += match_draws;
            chain_count++;
        }

        *out_len = chain_count;

    } else {
        // Normal analysis
        process_combinations(j, k, max_number, table, use_count,
                           m, results, l + n, out_len);
    }

    free_subset_table(table);

    if (*out_len == 0) {
        _mm_free(results);
        return NULL;
    }

    return results;
}

// Free results array
void free_analysis_results(AnalysisResultItem* results) {
    _mm_free(results);
}
