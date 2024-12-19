#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstring>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Получение текущего ранга процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Получение общего числа процессов
    vector<int> message_sizes = {1, 10, 100, 1000, 10000, 100000, 1000000};
    vector<int> exchange_counts = {10, 100, 1000, 10000};

    if (rank == 0) {
        cout << "Message size (bytes) | Number of exchanges | Average exchange time (sec)\n";
        cout << "--------------------------------------------------------------\n";
    }

    for (int message_size : message_sizes) {
        for (int num_exchanges : exchange_counts) {
            // Создаем буферы для отправки и приема
            char* send_buffer = new char[message_size];
            char* recv_buffer = new char[message_size];
            memset(send_buffer, 0, message_size);
            memset(recv_buffer, 0, message_size);

            MPI_Barrier(MPI_COMM_WORLD); // Синхронизация процессов перед началом измерений

            double start_time = MPI_Wtime(); // Начало измерения времени

            // Основной цикл обменов
            for (int i = 0; i < num_exchanges; ++i) {
                if (rank == 0) {
                    MPI_Send(send_buffer, message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(recv_buffer, message_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (rank == 1) {
                    MPI_Recv(recv_buffer, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(send_buffer, message_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                }
            }

            double end_time = MPI_Wtime(); // Конец измерения времени
            double total_time = end_time - start_time; // Общее время

            delete[] send_buffer; // Освобождение памяти
            delete[] recv_buffer;

            if (rank == 0) {
                cout << message_size << "                  | " 
                     << num_exchanges << "               | "
                     << total_time / num_exchanges << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
