#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int send_count = 4;
    std::vector<int> send_data(send_count * world_size); // Data in each process to send

    // Initialize send_data in each process
    for (int i = 0; i < send_count; i++) {
        send_data[i + world_rank * send_count] = world_rank * send_count + i;

        std::cout<<send_count<<" "<<world_rank<<" "<<i<<std::endl;
    }

    std::vector<int> recv_data(send_count * world_size); // Data to receive in root process

    // Gather data from all processes to root process
    MPI_Gather(send_data.data() + world_rank * send_count, send_count, MPI_INT, recv_data.data(), send_count, MPI_INT, 0, MPI_COMM_WORLD);

    // Print gathered data in root process
    if (world_rank == 0) {
        std::cout << "Root process gathered data: ";
        for (int i = 0; i < send_count * world_size; i++) {
            std::cout << recv_data[i] << " ";
        }
        std::cout << std::endl;
    }

    // Scatter data from root process back to all processes
    MPI_Scatter(recv_data.data(), send_count, MPI_INT, send_data.data() + world_rank * send_count, send_count, MPI_INT, 0, MPI_COMM_WORLD);

    // Print received data in each process after scattering
    std::cout << "Process " << world_rank << " received: ";
    for (int i = 0; i < send_count; i++) {
        std::cout << send_data[i + world_rank * send_count] << " ";
    }
    std::cout << std::endl;

    MPI_Finalize();
    return 0;
}
