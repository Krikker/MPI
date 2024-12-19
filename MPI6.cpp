#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <numeric>

using namespace std;

// Функция проверки результатов вычисления
bool verify_results(const vector<int>& result1, const vector<int>& result2, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (result1[i] != result2[i]) {
            return false;
        }
    }
    return true;
}

// Параллельное умножение матриц
void matrix_multiply_parallel(const vector<int>& A, const vector<int>& B, vector<int>& C,
                              int N, int rank, int size, const string& mode) {
    int block_size = N / size;

    if (N % size != 0) {
        if (rank == 0) {
            cerr << "Matrix size " << N << " is not divisible by number of processes " << size << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vector<int> local_A(block_size * N);
    vector<int> local_C(block_size * N, 0);

    int buffer_size = (block_size * N + MPI_BSEND_OVERHEAD) * sizeof(int);
    vector<char> buffer(buffer_size * size);

    if (mode == "buffered") {
        MPI_Buffer_attach(buffer.data(), buffer.size());
        if (rank == 0) {
            cout << "Buffer attached for mode: buffered" << endl;
        }
    }

    if (rank == 0) {
        // Отправка блоков матрицы A процессам
        for (int i = 1; i < size; ++i) {
            if (mode == "sync") {
                MPI_Ssend(A.data() + i * block_size * N, block_size * N, MPI_INT, i, 0, MPI_COMM_WORLD);
            } else if (mode == "ready") {
                MPI_Rsend(A.data() + i * block_size * N, block_size * N, MPI_INT, i, 0, MPI_COMM_WORLD);
            } else if (mode == "buffered") {
                MPI_Bsend(A.data() + i * block_size * N, block_size * N, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        local_A.assign(A.begin(), A.begin() + block_size * N);
    } else {
        MPI_Recv(local_A.data(), block_size * N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Распространение матрицы B всем процессам
    MPI_Bcast(const_cast<int*>(B.data()), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Вычисление локальной части результата
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    // Сборка результирующей матрицы
    MPI_Gather(local_C.data(), block_size * N, MPI_INT, C.data(), block_size * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (mode == "buffered") {
        void* detach_buffer;
        int detach_buffer_size;
        MPI_Buffer_detach(&detach_buffer, &detach_buffer_size);
        if (rank == 0) {
            cout << "Buffer detached for mode: buffered" << endl;
        }
    }
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
    vector<string> modes = {"sync", "ready", "buffered"};

    if (rank == 0) {
        cout << "Matrix Size | Transfer Mode | Number of Processes | Execution Time (sec) | Correctness\n";
        cout << "----------------------------------------------------------------------------------------\n";
    }

    for (int N : matrix_sizes) {
        for (const auto& mode : modes) {
            vector<int> A(N * N), B(N * N), C_seq(N * N, 0), C_parallel(N * N, 0);

            if (rank == 0) {
                srand(static_cast<unsigned>(time(0)));
                for (int i = 0; i < N * N; ++i) {
                    A[i] = rand() % 10;
                    B[i] = rand() % 10;
                }

                matrix_multiply_simple(A, B, C_seq, N);
            }

            auto start_time = chrono::high_resolution_clock::now();
            matrix_multiply_parallel(A, B, C_parallel, N, rank, size, mode);
            auto end_time = chrono::high_resolution_clock::now();
            chrono::duration<double> parallel_duration = end_time - start_time;
            double parallel_time = parallel_duration.count();

            if (rank == 0) {
                bool correct = verify_results(C_seq, C_parallel, N);

                cout << N << "            | " << mode << "           | " << size << "                | "
                     << parallel_time << "                 | " << (correct ? "Yes" : "No") << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
