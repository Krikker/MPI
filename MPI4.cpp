#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <numeric>
#include <cmath>

using namespace std;

constexpr double EPSILON = 1e-6;

// Функция для проверки корректности результатов
bool verify_results(const vector<int>& result1, const vector<int>& result2, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (abs(result1[i] - result2[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

// Параллельное умножение матриц
void matrix_multiply_parallel(const vector<int>& A, const vector<int>& B, vector<int>& C, int N, int rank, int size) {
    int block_size = N / size; // Размер блока для каждого процесса
    vector<int> local_A(block_size * N);
    vector<int> local_C(block_size * N, 0);

    // Распределение строк матрицы A между процессами
    MPI_Scatter(A.data(), block_size * N, MPI_INT, local_A.data(), block_size * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Широковещательная передача матрицы B всем процессам
    MPI_Bcast(const_cast<int*>(B.data()), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Локальное умножение блоков матриц
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    // Сбор результатов умножения от всех процессов
    MPI_Gather(local_C.data(), block_size * N, MPI_INT, C.data(), block_size * N, MPI_INT, 0, MPI_COMM_WORLD);
}

// Последовательное умножение матриц
void matrix_multiply_simple(const vector<int>& A, const vector<int>& B, vector<int>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение текущего ранга процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение общего числа процессов
    vector<int> matrix_sizes = {192, 384, 768};

    if (rank == 0) {
        cout << "Matrix size | Processes count | Parallel (s)  | Sequential (s) | Correctness\n";
        cout << "---------------------------------------------------------------------------\n";
    }

    for (int N : matrix_sizes) {
        vector<int> A(N * N), B(N * N), C_seq(N * N, 0), C_parallel(N * N, 0);

        // Инициализация матриц
        if (rank == 0) {
            srand(static_cast<unsigned>(time(0)));
            for (int i = 0; i < N * N; ++i) {
                A[i] = rand() % 10;
                B[i] = rand() % 10;
            }
        }

        // Последовательное умножение матриц
        double seq_start_time = MPI_Wtime();
        matrix_multiply_simple(A, B, C_seq, N);
        double seq_end_time = MPI_Wtime();
        double seq_time = seq_end_time - seq_start_time;

        // Параллельное умножение матриц
        double start_time = MPI_Wtime();
        matrix_multiply_parallel(A, B, C_parallel, N, rank, size);
        double end_time = MPI_Wtime();
        double parallel_time = end_time - start_time;

        bool parallel_correct = false;

        if (rank == 0) {
            parallel_correct = verify_results(C_seq, C_parallel, N);

            cout << N << "           | " << size << "               | "
                 << parallel_time << "        | "
                 << seq_time << "        | " << (parallel_correct ? "Yes" : "No") << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
