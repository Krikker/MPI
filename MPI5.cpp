#include <mpi.h>
#include <iostream>
#include <unistd.h>
#include <vector>
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
        cout << "Computation delay (us) | Message size (bytes) | Number of transfers | Execution time (s)\n";
        cout << "-------------------------------------------------------------------------------\n";
    }

    // Перебор всех комбинаций параметров
    for (int delay_time : computation_delays) {
        for (int message_size : message_sizes) {
            for (int transfers : num_transfers) {

                double start_time = MPI_Wtime(); // Начало измерения времени

                // Буферы для отправки и получения сообщений
                vector<char> send_buffer(message_size, rank);
                vector<char> recv_buffer(message_size, 0);

                if (rank == 0) {
                    // Процесс с рангом 0 отправляет сообщения всем остальным процессам
                    for (int i = 1; i < size; i++) {
                        MPI_Send(send_buffer.data(), message_size, MPI_CHAR, i, 0, MPI_COMM_WORLD);
                    }

                    // Процесс с рангом 0 принимает сообщения от всех процессов
                    for (int t = 0; t < transfers; t++) {
                        for (int i = 1; i < size; i++) {
                            MPI_Recv(recv_buffer.data(), message_size, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                    }
                } else {
                    // Остальные процессы получают сообщение от процесса 0
                    MPI_Recv(recv_buffer.data(), message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Имитация вычислений
                    do_computations(delay_time / size);

                    // Остальные процессы отправляют сообщение обратно процессу 0
                    for (int t = 0; t < transfers; t++) {
                        MPI_Send(send_buffer.data(), message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
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
