#include <mpi.h>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <chrono>

using namespace std;

// Функция, имитирующая вычисления с задержкой
void do_computations(int delay_time_us) {
    usleep(delay_time_us);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение текущего ранга процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение общего числа процессов
    vector<int> computation_delays = {1000, 10000, 100000, 1000000};
    vector<int> message_sizes = {1024, 10240, 102400, 1048576};
    vector<int> num_transfers = {1, 10, 100};

    if (rank == 0) {
        cout << "Computation Time (us) | Message Size (bytes) | Transfers Count | Execution Time (sec)\n";
        cout << "--------------------------------------------------------------------------\n";
    }

    // Циклы по всем комбинациям параметров
    for (int delay_time : computation_delays) {
        for (int message_size : message_sizes) {
            for (int transfers : num_transfers) {

                double start_time = MPI_Wtime(); // Начало измерения времени

                vector<char> send_buffer(message_size, rank);
                vector<char> recv_buffer(message_size, 0);

                MPI_Request request;

                if (rank == 0) {
                    // Процесс с рангом 0 отправляет данные остальным процессам
                    for (int i = 1; i < size; ++i) {
                        MPI_Isend(send_buffer.data(), message_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, &request);
                        MPI_Wait(&request, MPI_STATUS_IGNORE);
                    }

                    // Получение данных от других процессов
                    for (int t = 0; t < transfers; ++t) {
                        for (int i = 1; i < size; ++i) {
                            MPI_Irecv(recv_buffer.data(), message_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, &request);
                            MPI_Wait(&request, MPI_STATUS_IGNORE);
                        }
                    }
                } else {
                    // Остальные процессы получают данные от процесса с рангом 0
                    MPI_Irecv(recv_buffer.data(), message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &request);
                    do_computations(delay_time / size); // Выполнение вычислений
                    MPI_Wait(&request, MPI_STATUS_IGNORE);

                    // Отправка данных обратно процессу с рангом 0
                    for (int t = 0; t < transfers; ++t) {
                        MPI_Isend(send_buffer.data(), message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &request);
                        MPI_Wait(&request, MPI_STATUS_IGNORE);
                    }
                }

                double end_time = MPI_Wtime(); // Конец измерения времени

                if (rank == 0) {
                    cout << delay_time << "              | "
                         << message_size << "                | "
                         << transfers << "                 | "
                         << end_time - start_time << "              \n";
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
