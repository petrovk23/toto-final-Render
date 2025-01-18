#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "analysis_engine.h"

#define MAX_COMBO_STR 255
#define INITIAL_SUBSET_BUFFER 4096
#define SUBSET_BUFFER_INCREMENT 4096
#define MAX_NUMBERS 50
#define MAX_ALLOWED_J 200
#define HASH_SIZE (1 << 24)  // 16M entries
#define MAX_TOP_RESULTS 5000
#define MAX_SELECTED_RESULTS 1000

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

typedef struct {
    char* data;
    size_t capacity;
    size_t length;
} DynamicString;

static void set_optimal_threads() {
    int max_available = omp_get_max_threads();
    omp_set_num_threads(max_available);
}

static void init_tables() {
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
        for (int b = 0; b < 8; b++) {
            if (i & (1 << b)) c++;
        }
        bit_count_table[i] = c;
    }
    initialized = 1;
}

static inline int popcount64(uint64 x) {
    int res = 0;
    for (int i = 0; i < 8; i++) {
        res += bit_count_table[x & 0xFF];
        x >>= 8;
    }
    return res;
}

static DynamicString* create_dynamic_string() {
    DynamicString* str = (DynamicString*)malloc(sizeof(DynamicString));
    if (!str) return NULL;

    str->data = (char*)malloc(INITIAL_SUBSET_BUFFER);
    if (!str->data) {
        free(str);
        return NULL;
    }

    str->capacity = INITIAL_SUBSET_BUFFER;
    str->length = 0;
    str->data[0] = '\0';
    return str;
}

static int dynamic_string_append(DynamicString* str, const char* text) {
    size_t len = strlen(text);
    size_t new_length = str->length + len;

    if (new_length >= str->capacity) {
        size_t new_capacity = str->capacity;
        while (new_capacity <= new_length) {
            new_capacity += SUBSET_BUFFER_INCREMENT;
        }

        char* new_data = (char*)realloc(str->data, new_capacity);
        if (!new_data) return 0;

        str->data = new_data;
        str->capacity = new_capacity;
    }

    strcpy(str->data + str->length, text);
    str->length = new_length;
    return 1;
}

static void free_dynamic_string(DynamicString* str) {
    if (str) {
        free(str->data);
        free(str);
    }
}

static SubsetTable* create_subset_table(int max_entries) {
    SubsetTable* t = (SubsetTable*)malloc(sizeof(SubsetTable));
    if (!t) return NULL;
    t->size = 0;
    t->capacity = max_entries;
    t->keys = (uint64*)calloc(max_entries, sizeof(uint64));
    t->values = (int*)malloc(max_entries * sizeof(int));
    if (!t->keys || !t->values) {
        free(t->keys);
        free(t->values);
        free(t);
        return NULL;
    }
    for (int i = 0; i < max_entries; i++) {
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

static inline uint32 hash_subset(uint64 pattern) {
    pattern ^= pattern >> 33;
    pattern *= 0xff51afd7ed558ccdULL;
    pattern ^= pattern >> 33;
    pattern *= 0xc4ceb9fe1a85ec53ULL;
    pattern ^= pattern >> 33;
    return (uint32)(pattern & (HASH_SIZE - 1));
}

static inline void insert_subset(SubsetTable* table, uint64 pattern, int value) {
    uint32 idx = hash_subset(pattern);
    while (1) {
        if (table->values[idx] == -1 || table->keys[idx] == pattern) {
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
        if (table->values[idx] == -1) return -1;
        if (table->keys[idx] == pattern) return table->values[idx];
        idx = (idx + 1) & (HASH_SIZE - 1);
    }
}

static inline uint64 numbers_to_pattern(const int* numbers, int count) {
    uint64 p = 0ULL;
    for (int i = 0; i < count; i++) {
        p |= (1ULL << (numbers[i] - 1));
    }
    return p;
}

static void process_draw(const int* draw, int draw_idx, int k, SubsetTable* table) {
    if (k > 6) return;
    int idx[MAX_NUMBERS];
    for (int i = 0; i < k; i++) {
        idx[i] = i;
    }
    while (1) {
        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) {
            pat |= (1ULL << (draw[idx[i]] - 1));
        }
        insert_subset(table, pat, draw_idx);

        int pos = k - 1;
        while (pos >= 0 && idx[pos] == 6 - k + pos) pos--;
        if (pos < 0) break;
        idx[pos]++;
        for (int x = pos + 1; x < k; x++) {
            idx[x] = idx[pos] + (x - pos);
        }
    }
}

static void evaluate_combo(const int* combo, int j, int k, int total_draws,
                           const SubsetTable* table, ComboStats* stats) {
    double sum_ranks = 0.0;
    double min_rank = (double)total_draws;
    int count = 0;

    int idx[MAX_NUMBERS];
    for (int i = 0; i < k; i++) {
        idx[i] = i;
    }

    while (1) {
        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) {
            pat |= (1ULL << (combo[idx[i]] - 1));
        }
        int last_seen = lookup_subset(table, pat);
        double rank = (last_seen >= 0)
                      ? (double)(total_draws - last_seen - 1)
                      : (double)total_draws;
        sum_ranks += rank;
        if (rank < min_rank) {
            min_rank = rank;
        }
        count++;

        int pos = k - 1;
        while (pos >= 0 && idx[pos] == j - k + pos) pos--;
        if (pos < 0) break;
        idx[pos]++;
        for (int x = pos + 1; x < k; x++) {
            idx[x] = idx[pos] + (x - pos);
        }
    }

    stats->pattern = numbers_to_pattern(combo, j);
    stats->avg_rank = sum_ranks / (double)count;
    stats->min_rank = min_rank;
    memcpy(stats->combo, combo, j * sizeof(int));
    stats->len = j;
}

static void format_combo(const int* combo, int len, char* out) {
    int pos = 0;
    for (int i = 0; i < len; i++) {
        if (i > 0) {
            out[pos++] = ',';
            out[pos++] = ' ';
        }
        pos += sprintf(out + pos, "%d", combo[i]);
    }
    out[pos] = '\0';
}

static char* format_subsets_dynamic(const int* combo, int j, int k, int total_draws,
                                  const SubsetTable* table)
{
    DynamicString* dstr = create_dynamic_string();
    if (!dstr) return NULL;

    dynamic_string_append(dstr, "[");

    int idx[MAX_NUMBERS];
    for (int i = 0; i < k; i++) {
        idx[i] = i;
    }
    int first = 1;
    char temp[64];

    while (1) {
        if (!first) {
            dynamic_string_append(dstr, ", ");
        }
        first = 0;

        dynamic_string_append(dstr, "((");

        for (int i = 0; i < k; i++) {
            if (i > 0) {
                dynamic_string_append(dstr, ", ");
            }
            sprintf(temp, "%d", combo[idx[i]]);
            dynamic_string_append(dstr, temp);
        }
        dynamic_string_append(dstr, "), ");

        uint64 pat = 0ULL;
        for (int i = 0; i < k; i++) {
            pat |= (1ULL << (combo[idx[i]] - 1));
        }
        int last_seen = lookup_subset(table, pat);
        int rank = (last_seen >= 0)
                   ? (total_draws - last_seen - 1)
                   : total_draws;

        sprintf(temp, "%d)", rank);
        dynamic_string_append(dstr, temp);

        int p = k - 1;
        while (p >= 0 && idx[p] == j - k + p) p--;
        if (p < 0) break;
        idx[p]++;
        for (int x = p + 1; x < k; x++) {
            idx[x] = idx[p] + (x - p);
        }
    }

    dynamic_string_append(dstr, "]");

    char* result = strdup(dstr->data);
    free_dynamic_string(dstr);
    return result;
}

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
) {
    SubsetTable* table = create_subset_table(HASH_SIZE);
    if (!table) return NULL;

    for (int i = 0; i < use_count; i++) {
        process_draw(&sorted_draws_data[i * 6], i, k, table);
    }

    int actual_l = (l > MAX_TOP_RESULTS) ? MAX_TOP_RESULTS : l;
    int actual_n = (n > MAX_SELECTED_RESULTS) ? MAX_SELECTED_RESULTS : n;

    ComboStats* best_stats = (ComboStats*)malloc(actual_l * sizeof(ComboStats));
    if (!best_stats) {
        free_subset_table(table);
        return NULL;
    }
    memset(best_stats, 0, actual_l * sizeof(ComboStats));

    AnalysisResultItem* results = (AnalysisResultItem*)calloc(actual_l + actual_n, sizeof(AnalysisResultItem));
    if (!results) {
        free(best_stats);
        free_subset_table(table);
        return NULL;
    }

    int filled = 0;
    int error_occurred = 0;

    #pragma omp parallel
    {
        int* curr_combo = NULL;
        ComboStats* thread_best = NULL;
        int thread_filled = 0;

        curr_combo = (int*)malloc(j * sizeof(int));
        if (!curr_combo) {
            #pragma omp atomic write
            error_occurred = 1;
        } else {
            thread_best = (ComboStats*)malloc(actual_l * sizeof(ComboStats));
            if (!thread_best) {
                free(curr_combo);
                #pragma omp atomic write
                error_occurred = 1;
            } else {
                memset(thread_best, 0, actual_l * sizeof(ComboStats));

                #pragma omp for schedule(dynamic)
                for (int first = 1; first <= max_number - j + 1; first++) {
                    if (!error_occurred) {
                        curr_combo[0] = first;
                        for (int i = 1; i < j; i++) {
                            curr_combo[i] = first + i;
                        }

                        while (1) {
                            ComboStats stats;
                            evaluate_combo(curr_combo, j, k, use_count, table, &stats);

                            if (thread_filled < actual_l) {
                                memcpy(&thread_best[thread_filled], &stats, sizeof(ComboStats));
                                thread_filled++;
                                for (int i = thread_filled - 1; i > 0; i--) {
                                    int swap;
                                    if (strcmp(m, "avg") == 0) {
                                        swap = (thread_best[i].avg_rank > thread_best[i - 1].avg_rank);
                                    } else {
                                        swap = (thread_best[i].min_rank > thread_best[i - 1].min_rank);
                                    }
                                    if (swap) {
                                        ComboStats tmp = thread_best[i];
                                        thread_best[i] = thread_best[i - 1];
                                        thread_best[i - 1] = tmp;
                                    } else {
                                        break;
                                    }
                                }
                            } else {
                                double val = (strcmp(m, "avg") == 0) ? stats.avg_rank : stats.min_rank;
                                double worst_val = (strcmp(m, "avg") == 0)
                                    ? thread_best[actual_l - 1].avg_rank
                                    : thread_best[actual_l - 1].min_rank;
                                if (val > worst_val) {
                                    thread_best[actual_l - 1] = stats;
                                    for (int i = actual_l - 1; i > 0; i--) {
                                        double vcur = (strcmp(m, "avg") == 0)
                                            ? thread_best[i].avg_rank
                                            : thread_best[i].min_rank;
                                        double vprev = (strcmp(m, "avg") == 0)
                                            ? thread_best[i - 1].avg_rank
                                            : thread_best[i - 1].min_rank;
                                        if (vcur > vprev) {
                                            ComboStats tmp = thread_best[i];
                                            thread_best[i] = thread_best[i - 1];
                                            thread_best[i - 1] = tmp;
                                        } else {
                                            break;
                                        }
                                    }
                                }
                            }

                            int pos = j - 1;
                            while (pos >= 0 && curr_combo[pos] == max_number - j + pos + 1) pos--;
                            if (pos < 0 || pos == 0) break;
                            curr_combo[pos]++;
                            for (int x = pos + 1; x < j; x++) {
                                curr_combo[x] = curr_combo[pos] + (x - pos);
                            }
                        }
                    }
                }

                #pragma omp critical
                {
                    for (int i = 0; i < thread_filled; i++) {
                        if (filled < actual_l) {
                            memcpy(&best_stats[filled], &thread_best[i], sizeof(ComboStats));
                            filled++;
                        } else {
                            double val = (strcmp(m, "avg") == 0)
                                ? thread_best[i].avg_rank
                                : thread_best[i].min_rank;
                            double worst_val = (strcmp(m, "avg") == 0)
                                ? best_stats[actual_l - 1].avg_rank
                                : best_stats[actual_l - 1].min_rank;
                            if (val > worst_val) {
                                best_stats[actual_l - 1] = thread_best[i];
                                for (int idx = actual_l - 1; idx > 0; idx--) {
                                    double vcur = (strcmp(m, "avg") == 0)
                                        ? best_stats[idx].avg_rank
                                        : best_stats[idx].min_rank;
                                    double vprev = (strcmp(m, "avg") == 0)
                                        ? best_stats[idx - 1].avg_rank
                                        : best_stats[idx - 1].min_rank;
                                    if (vcur > vprev) {
                                        ComboStats tmp = best_stats[idx];
                                        best_stats[idx] = best_stats[idx - 1];
                                        best_stats[idx - 1] = tmp;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (thread_best) free(thread_best);
        if (curr_combo) free(curr_combo);
    }

    if (error_occurred) {
        free(best_stats);
        free(results);
        free_subset_table(table);
        return NULL;
    }

    int top_count = (filled < actual_l) ? filled : actual_l;
    int results_count = 0;

    free_subset_table(table);
    table = create_subset_table(HASH_SIZE);
    for (int i = 0; i < use_count; i++) {
        process_draw(&sorted_draws_data[i * 6], i, k, table);
    }

    for (int i = 0; i < top_count; i++) {
        format_combo(best_stats[i].combo, best_stats[i].len, results[results_count].combination);
        results[results_count].subsets = format_subsets_dynamic(
            best_stats[i].combo, j, k, use_count, table
        );
        results[results_count].avg_rank = best_stats[i].avg_rank;
        results[results_count].min_value = best_stats[i].min_rank;
        results[results_count].is_chain_result = 0;
        results[results_count].draw_offset = 0;
        results[results_count].analysis_start_draw = 0;
        results[results_count].draws_until_common = 0;
        results_count++;
    }

    int second_table_count = 0;
    int* pick_indices = NULL;
    if (actual_n > 0 && top_count > 0) {
        pick_indices = (int*)malloc(top_count * sizeof(int));
        memset(pick_indices, -1, top_count * sizeof(int));

        int chosen = 0;
        pick_indices[chosen++] = 0;

        for (int i = 1; i < top_count && chosen < actual_n; i++) {
            uint64 pat_i = best_stats[i].pattern;
            int overlap = 0;
            for (int c = 0; c < chosen; c++) {
                int idxC = pick_indices[c];
                uint64 pat_c = best_stats[idxC].pattern;
                uint64 inter = (pat_i & pat_c);
                if (popcount64(inter) >= k) {
                    overlap = 1;
                    break;
                }
            }
            if (!overlap) {
                pick_indices[chosen++] = i;
            }
        }
        second_table_count = chosen;
    }

    int bottom_start = results_count;
    for (int i = 0; i < second_table_count; i++) {
        int idx = pick_indices[i];
        format_combo(best_stats[idx].combo, best_stats[idx].len, results[bottom_start + i].combination);
        results[bottom_start + i].subsets = format_subsets_dynamic(
            best_stats[idx].combo, j, k, use_count, table
        );
        results[bottom_start + i].avg_rank = best_stats[idx].avg_rank;
        results[bottom_start + i].min_value = best_stats[idx].min_rank;
        results[bottom_start + i].is_chain_result = 0;
        results[bottom_start + i].draw_offset = 0;
        results[bottom_start + i].analysis_start_draw = 0;
        results[bottom_start + i].draws_until_common = 0;
    }
    int total_used = results_count + second_table_count;
    *out_len = total_used;

    if (pick_indices) {
        free(pick_indices);
    }
    free_subset_table(table);
    free(best_stats);

    if (total_used == 0) {
        free(results);
        return NULL;
    }
    return results;
}

static AnalysisResultItem* run_chain_analysis(
    const int* sorted_draws_data,
    int draws_count,
    int initial_offset,
    int j,
    int k,
    const char* m,
    int max_number,
    int* out_len
) {
    AnalysisResultItem* chain_results = (AnalysisResultItem*)calloc(initial_offset + 2, sizeof(AnalysisResultItem));
    if (!chain_results) {
        *out_len = 0;
        return NULL;
    }

    uint64* draw_patterns = (uint64*)malloc(draws_count * sizeof(uint64));
    if (!draw_patterns) {
        free(chain_results);
        *out_len = 0;
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        uint64 pat = 0ULL;
        for (int z = 0; z < 6; z++) {
            pat |= (1ULL << (sorted_draws_data[i * 6 + z] - 1));
        }
        draw_patterns[i] = pat;
    }

    int chain_index = 0;
    int current_offset = initial_offset;

    while (1) {
        if (current_offset < 0) {
            break;
        }
        if (current_offset > (draws_count - 1)) {
            break;
        }
        int use_count = draws_count - current_offset;
        if (use_count < 1) {
            break;
        }

        SubsetTable* table = create_subset_table(HASH_SIZE);
        for (int i = 0; i < use_count; i++) {
            process_draw(&sorted_draws_data[i * 6], i, k, table);
        }

        int found_any = 0;
        double best_val = -1e9;
        ComboStats best_stat;
        memset(&best_stat, 0, sizeof(best_stat));

        int* combo_buf = (int*)malloc(j * sizeof(int));
        if (!combo_buf) {
            free_subset_table(table);
            break;
        }
        for (int i = 0; i < j; i++) {
            combo_buf[i] = i + 1;
        }

        while (1) {
            ComboStats stats;
            evaluate_combo(combo_buf, j, k, use_count, table, &stats);

            double val = (strcmp(m, "avg") == 0) ? stats.avg_rank : stats.min_rank;
            if (!found_any || val > best_val) {
                best_val = val;
                best_stat = stats;
                found_any = 1;
            }

            int pos = j - 1;
            while (pos >= 0 && combo_buf[pos] == max_number - j + pos + 1) pos--;
            if (pos < 0) break;
            combo_buf[pos]++;
            for (int x = pos + 1; x < j; x++) {
                combo_buf[x] = combo_buf[pos] + (x - pos);
            }
        }
        free(combo_buf);
        free_subset_table(table);

        if (!found_any) {
            break;
        }

        AnalysisResultItem* out_item = &chain_results[chain_index];
        format_combo(best_stat.combo, best_stat.len, out_item->combination);

        SubsetTable* tmp_t = create_subset_table(HASH_SIZE);
        for (int i = 0; i < use_count; i++) {
            process_draw(&sorted_draws_data[i * 6], i, k, tmp_t);
        }
        out_item->subsets = format_subsets_dynamic(
            best_stat.combo, j, k, use_count, tmp_t
        );
        free_subset_table(tmp_t);

        out_item->avg_rank = best_stat.avg_rank;
        out_item->min_value = best_stat.min_rank;
        out_item->is_chain_result = 1;
        out_item->draw_offset = chain_index + 1;
        out_item->analysis_start_draw = draws_count - current_offset;

        uint64 combo_pat = 0ULL;
        for (int z = 0; z < j; z++) {
            combo_pat |= (1ULL << (best_stat.combo[z] - 1));
        }

        int found_common = 0;
        int i;
        for (i = 1; i <= current_offset; i++) {
            int f_idx = draws_count - 1 - (current_offset - i);
            if (f_idx < 0) break;
            uint64 fpat = draw_patterns[f_idx];
            uint64 inter = (combo_pat & fpat);
            if (popcount64(inter) >= k) {
                found_common = 1;
                break;
            }
        }
        if (!found_common) {
            i = current_offset + 1;
        } else if (i > current_offset) {
            i = current_offset;
        }

        out_item->draws_until_common = (i > 0) ? (i - 1) : 0;
        current_offset -= i;
        chain_index++;

        if (current_offset <= 0) {
            break;
        }
    }

    free(draw_patterns);
    *out_len = chain_index;
    if (chain_index == 0) {
        free(chain_results);
        return NULL;
    }
    return chain_results;
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
    *out_len = 0;
    if (j > MAX_ALLOWED_J) {
        return NULL;
    }
    init_tables();
    set_optimal_threads();

    int max_number = (strstr(game_type, "6_49")) ? 49 : 42;
    if (draws_count < 1) {
        return NULL;
    }

    int* sorted_draws_data = (int*)malloc(draws_count * 6 * sizeof(int));
    if (!sorted_draws_data) {
        return NULL;
    }
    for (int i = 0; i < draws_count; i++) {
        int temp[6];
        for (int z = 0; z < 6; z++) {
            temp[z] = draws[i][z];
        }
        for (int a = 0; a < 5; a++) {
            for (int b = a + 1; b < 6; b++) {
                if (temp[a] > temp[b]) {
                    int t = temp[a];
                    temp[a] = temp[b];
                    temp[b] = t;
                }
            }
        }
        for (int z = 0; z < 6; z++) {
            sorted_draws_data[i * 6 + z] = temp[z];
        }
    }

    AnalysisResultItem* results;
    if (l != -1) {
        int use_count = draws_count - last_offset;
        if (use_count < 1) {
            free(sorted_draws_data);
            return NULL;
        }
        results = run_standard_analysis(
            sorted_draws_data,
            use_count,
            j, k, m, l, n, max_number,
            out_len
        );
    } else {
        results = run_chain_analysis(
            sorted_draws_data,
            draws_count,
            last_offset,
            j, k, m,
            max_number,
            out_len
        );
    }

    free(sorted_draws_data);
    return results;
}

void free_analysis_results(AnalysisResultItem* results, int* out_len) {
    if (results) {
        for (int i = 0; i < *out_len; i++) {
            if (results[i].subsets) {
                free(results[i].subsets);
            }
        }
        free(results);
    }
}
