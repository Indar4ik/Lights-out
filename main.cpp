#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <print>
#include <format>


constexpr int MAX_N = 1535;
constexpr int MAX_WORDS = (MAX_N + 64) / 64; // 24 слова (1536 бит)

// SIMD битовый вектор
struct alignas(32) Vec {
    uint64_t d[MAX_WORDS];

    void set_zero() {
        #pragma unroll
        for (int i = 0; i < MAX_WORDS; ++i) d[i] = 0;
    }

    void assign_xor(const Vec& a, const Vec& b, const Vec& c, const Vec& e) {
        #pragma ivdep
        #pragma unroll
        for (int i = 0; i < MAX_WORDS; ++i) {
            d[i] = a.d[i] ^ b.d[i] ^ c.d[i] ^ e.d[i];
        }
    }

    void flip_bit(int bit) { d[bit >> 6] ^= (1ULL << (bit & 63)); }
    int test_bit(int bit) const { return (d[bit >> 6] >> (bit & 63)) & 1; }

    Vec shift_left() const {
        Vec res;
        uint64_t carry = 0;
        // #pragma clang loop unroll(full)
        #pragma unroll
        for (int i = 0; i < MAX_WORDS; ++i) {
            res.d[i] = (d[i] << 1) | carry;
            carry = d[i] >> 63;
        }
        return res;
    }

    Vec shift_right() const {
        Vec res;
        uint64_t carry = 0;
        #pragma unroll
        for (int i = MAX_WORDS - 1; i >= 0; --i) {
            res.d[i] = (d[i] >> 1) | carry;
            carry = d[i] << 63;
        }
        return res;
    }

    void flip_all(int n) {
        int full_words = n / 64;
        int rem = n % 64;
        for (int i = 0; i < full_words; ++i) d[i] ^= ~0ULL;
        if (rem > 0) d[full_words] ^= (1ULL << rem) - 1;
    }

    void mask_to_n(int n) {
        int full_words = n / 64;
        int rem = n % 64;
        if (rem > 0) d[full_words] &= (1ULL << rem) - 1;
        for (int i = full_words + (rem > 0 ? 1 : 0); i < MAX_WORDS; ++i) d[i] = 0;
    }

    int count_ones(int n) const {
        int count = 0;
        int full_words = n / 64;
        int rem = n % 64;
        for(int i = 0; i < full_words; ++i) count += std::popcount(d[i]);
        if (rem > 0) count += std::popcount(d[full_words] & ((1ULL << rem) - 1));
        return count;
    }
};

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int a, b;
    if (!(std::cin >> a >> b)) return 0;

    std::string line;
    std::getline(std::cin, line);

    bool flag_M = false, flag_I = false, flag_O = false, flag_T = false, flag_S = false;
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        for (char c : token) {
            if (c == 'M') flag_M = true;
            if (c == 'I') flag_I = true;
            if (c == 'O') flag_O = true;
            if (c == 'T') flag_T = true;
            if (c == 'S') flag_S = true;
        }
    }

    Vec zero_vec;
    zero_vec.set_zero();

    #pragma omp parallel for schedule(dynamic, 1) ordered
    for (int n = a; n <= b; ++n) {
        [[assume(n >= 1 && n <= MAX_N)]];

        auto solve_start = std::chrono::steady_clock::now();

        // Пре-аллокация векторов
        std::vector<Vec> P_prev(n, zero_vec);
        std::vector<Vec> P_cur(n, zero_vec);
        std::vector<Vec> P_next(n, zero_vec);

        for (int j = 0; j < n; ++j) {
            P_cur[j].set_zero();
            P_cur[j].flip_bit(j);
        }

        // Вычисляем зависимости для всех строк вниз
        for (int i = 0; i < n; ++i) {
            if (n == 1) {
                P_next[0].assign_xor(P_cur[0], zero_vec, zero_vec, P_prev[0]);
                P_next[0].flip_bit(1);
            } else {
                // Крайний левый элемент (j = 0)
                P_next[0].assign_xor(P_cur[0], zero_vec, P_cur[1], P_prev[0]);
                P_next[0].flip_bit(n);
                
                for (int j = 1; j < n - 1; ++j) {
                    P_next[j].assign_xor(P_cur[j], P_cur[j - 1], P_cur[j + 1], P_prev[j]);
                    P_next[j].flip_bit(n);
                }
                
                // Крайний правый элемент (j = n - 1)
                P_next[n - 1].assign_xor(P_cur[n - 1], P_cur[n - 2], zero_vec, P_prev[n - 1]);
                P_next[n - 1].flip_bit(n);
            }
            std::swap(P_prev, P_cur);
            std::swap(P_cur, P_next);
        }

        auto& mat = P_cur;
        int rank = 0;
        int free_vars = 0;
        std::vector<int> pivot_row(n, -1);

        // Гауссово Исключение GF(2)
        for (int j = 0; j < n; ++j) {
            [[assume(j >= 0 && j < MAX_N)]];
            
            int pivot = -1;
            for (int i = rank; i < n; ++i) {
                if (mat[i].test_bit(j)) {
                    pivot = i;
                    break;
                }
            }
            if (pivot != -1) {
                std::swap(mat[rank], mat[pivot]);
                for (int i = 0; i < n; ++i) {
                    if (i != rank && mat[i].test_bit(j)) {
                        #pragma ivdep
                        #pragma unroll
                        for(int k = 0; k < MAX_WORDS; ++k) {
                            mat[i].d[k] ^= mat[rank].d[k];
                        }
                    }
                }
                pivot_row[j] = rank;
                rank++;
            } else {
                free_vars++;
            }
        }

        // Бит-параллельное восстановление матрицы
        std::vector<Vec> board(n, zero_vec);
        
        // 0-я строка
        for (int j = 0; j < n; ++j) {
            if (pivot_row[j] != -1 && mat[pivot_row[j]].test_bit(n)) {
                board[0].flip_bit(j);
            }
        }

        int ones_count = board[0].count_ones(n);

        // Вся остальная матрица векторными операциями
        for (int i = 0; i < n - 1; ++i) {
            Vec left = board[i].shift_left();
            Vec right = board[i].shift_right();
            const Vec& prev = (i > 0) ? board[i - 1] : zero_vec;
            
            board[i + 1].assign_xor(board[i], left, right, prev);
            board[i + 1].flip_all(n);
            board[i + 1].mask_to_n(n);
            
            ones_count += board[i + 1].count_ones(n);
        }

        auto solve_end = std::chrono::steady_clock::now();
        double solve_ms = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();

        double img_ms = 0.0;
        if (flag_I) {
            auto img_start = std::chrono::steady_clock::now();
            
            int row_bytes = (n + 7) / 8;
            std::vector<uint8_t> pbm_data(row_bytes * n, 0);
            
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (board[i].test_bit(j)) {
                        // Старший бит (MSB) = левый пиксель в байте (формат PBM)
                        pbm_data[i * row_bytes + (j / 8)] |= (1 << (7 - (j % 8)));
                    }
                }
            }

            std::string filename = std::format("images/solution{}.pbm", n);
            std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
            
            // Заголовок PBM (P4 = бинарный формат)
            std::string header = std::format("P4\n{} {}\n", n, n);
            ofs.write(header.c_str(), header.size());
            ofs.write(reinterpret_cast<const char*>(pbm_data.data()), pbm_data.size());
            
            auto img_end = std::chrono::steady_clock::now();
            img_ms = std::chrono::duration<double, std::milli>(img_end - img_start).count();
        }

        std::string out = std::format("n={}: ", n);
        if (flag_T) {
            out += std::format("solve={:.4f} ms ", solve_ms);
            if (flag_I) out += std::format("image={:.4f} ms", img_ms);
        }
        if (flag_O) out += std::format("ones={} ", ones_count);
        if (flag_S) out += std::format("solutions=2^{} ", free_vars);
        
        if (flag_M) {
            out += "Matrix:\n";
            out.reserve(out.size() + n * n * 2); 
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    out += (board[i].test_bit(j) ? '1' : '0');
                    out += (j == n - 1 ? '\n' : ' ');
                }
            }
        }

        #pragma omp ordered
        {
            std::print("{}\n", out);
        }
    }

    return 0;
}