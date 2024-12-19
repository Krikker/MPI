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

    // Проверка: программа должна быть запущена на 2 процессах
    if (size != 2) {
        if (rank == 0) {
            cerr << "The program must be run with 2 processes.\n";
        }
        MPI_Finalize();
        return 1;
    }

    vector<int> message_sizes = {1, 10, 100, 1000, 10000, 100000, 1000000};
    vector<int> exchange_counts = {10, 100, 1000, 10000};

    if (rank == 0) {
        cout << "Message size (bytes) | Number of exchanges | Avg exchange time (seconds)\n";
        cout << "-------------------------------------------------------------\n";
    }

    // Цикл по всем размерам сообщений
    for (int message_size : message_sizes) {
        // Цикл по количеству обменов
        for (int num_exchanges : exchange_counts) {
            // Выделение памяти для буферов отправки и приёма
            char* send_buffer = new char[message_size];
            char* recv_buffer = new char[message_size];
            memset(send_buffer, 0, message_size);
            memset(recv_buffer, 0, message_size);

            MPI_Barrier(MPI_COMM_WORLD); // Синхронизация всех процессов перед началом замера времени

            double start_time = MPI_Wtime(); // Начало измерения времени

            // Цикл обмена сообщениями между процессами
            for (int i = 0; i < num_exchanges; ++i) {
                if (rank == 0) {
                    // Процесс 0 отправляет и получает сообщение
                    MPI_Sendrecv(send_buffer, message_size, MPI_CHAR, 1, 0,
                                 recv_buffer, message_size, MPI_CHAR, 1, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (rank == 1) {
                    // Процесс 1 отправляет и получает сообщение
                    MPI_Sendrecv(send_buffer, message_size, MPI_CHAR, 0, 0,
                                 recv_buffer, message_size, MPI_CHAR, 0, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            double end_time = MPI_Wtime(); // Конец измерения времени
            double total_time = end_time - start_time;

            // Освобождение памяти для буферов
            delete[] send_buffer;
            delete[] recv_buffer;

            if (rank == 0) {
                cout << message_size << "                     | "
                     << num_exchanges << "             | "
                     << total_time / num_exchanges << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
