#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение текущего ранга процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение общего числа процессов
    vector<int> vector_sizes = {1000, 10000, 100000, 1000000, 10000000};

    if (rank == 0) {
        cout << "Vector Size | Number of Processes | Sequential Time | Parallel Time | Min | Max\n";
        cout << "-------------------------------------------------------------------------------------\n";
    }

    // Проходим по каждому размеру вектора
    for (int N : vector_sizes) {
        vector<int> data;
        int local_size = N / size;
        vector<int> local_data(local_size);
        double seq_start_time = 0.0, seq_end_time = 0.0;

        // Генерация данных и последовательное выполнение на нулевом процессе
        if (rank == 0) {
            data.resize(N);
            srand(static_cast<unsigned>(time(0)));
            for (int i = 0; i < N; ++i) {
                data[i] = rand() % 1000; // Заполнение случайными числами от 0 до 999
            }

            // Измерение времени последовательного выполнения
            seq_start_time = MPI_Wtime();
            int seq_min = *min_element(data.begin(), data.end());
            int seq_max = *max_element(data.begin(), data.end());
            seq_end_time = MPI_Wtime();
        }

        // Начало измерения времени параллельного выполнения
        double par_start_time = MPI_Wtime();

        // Рассылка данных локальным процессам
        MPI_Scatter(data.data(), local_size, MPI_INT, 
                    local_data.data(), local_size, MPI_INT, 
                    0, MPI_COMM_WORLD);

        // Локальный поиск минимума и максимума
        int local_min = *min_element(local_data.begin(), local_data.end());
        int local_max = *max_element(local_data.begin(), local_data.end());

        // Сбор глобальных минимумов и максимумов
        int global_min, global_max;
        MPI_Reduce(&local_min, &global_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

        // Конец измерения времени параллельного выполнения
        double par_end_time = MPI_Wtime();

        if (rank == 0) {
            cout << N << "           | "
                 << size << "                | "
                 << seq_end_time - seq_start_time << "                | "
                 << par_end_time - par_start_time << "              | "
                 << global_min << "  | "
                 << global_max << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
