#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <numeric>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение текущего ранга процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение общего числа процессов
    vector<int> vector_sizes = {1000, 10000, 100000, 1000000, 10000000};

    if (rank == 0) {
        cout << "Vector size | Number of processes | Sequential time | Parallel time | Result\n";
        cout << "-------------------------------------------------------------------------------\n";
    }

    // Цикл по различным размерам векторов
    for (int N : vector_sizes) {
        vector<int> vec1, vec2;
        int local_size = N / size; // Размер части вектора для каждого процесса
        vector<int> local_vec1(local_size), local_vec2(local_size);

        double seq_start_time = 0.0, seq_end_time = 0.0;
        long long scalar_result_seq = 0;

        // Инициализация векторов и вычисление скалярного произведения последовательно
        if (rank == 0) {
            vec1.resize(N);
            vec2.resize(N);
            srand(static_cast<unsigned>(time(0))); // Инициализация генератора случайных чисел
            for (int i = 0; i < N; ++i) {
                vec1[i] = rand() % 100;
                vec2[i] = rand() % 100;
            }

            // Измерение времени последовательного выполнения
            seq_start_time = MPI_Wtime();
            scalar_result_seq = inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0LL);
            seq_end_time = MPI_Wtime();
        }

        // Начало измерения времени параллельного выполнения
        double par_start_time = MPI_Wtime();

        // Распределяем части векторов по процессам
        MPI_Scatter(vec1.data(), local_size, MPI_INT, local_vec1.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Scatter(vec2.data(), local_size, MPI_INT, local_vec2.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

        // Каждый процесс вычисляет частичное скалярное произведение
        long long local_result = inner_product(local_vec1.begin(), local_vec1.end(), local_vec2.begin(), 0LL);

        // Суммируем результаты от всех процессов
        long long global_result = 0;
        MPI_Reduce(&local_result, &global_result, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        // Конец измерения времени параллельного выполнения
        double par_end_time = MPI_Wtime();

        if (rank == 0) {
            cout << N << "           | "
                 << size << "                | "
                 << seq_end_time - seq_start_time << "                | "
                 << par_end_time - par_start_time << "              | "
                 << global_result << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
