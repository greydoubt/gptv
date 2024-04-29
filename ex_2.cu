// std::thread, std::atomic, and std::barrier
// data exchange with neighbors and is split into three steps:
// internal
// prev_boundary
// next_boundary

The internal function processes internal rows that do not depend on data from neighbors.

double internal(double* u_new, double* u_old, parameters p) {
    grid g { .x_start = 2, .x_end = p.nx, .y_start = 1, .y_end = p.ny - 1 };
    energy += stencil(u_new.get(), u_old.get(), g, p);
}

// prev_boundary function exchanges data with neighbor at rank - 1 and processes the rows that depend on the elements received.

double prev_boundary(double* u_new, double* u_old, parameters p) {
    // Send window cells, receive halo cells
    if (p.rank > 0) {
      // Send bottom boundary to bottom rank
      MPI_Send(u_old + p.ny, p.ny, MPI_DOUBLE, p.rank - 1, 0, MPI_COMM_WORLD);
      // Receive top boundary from bottom rank
      MPI_Recv(u_old + 0, p.ny,  MPI_DOUBLE, p.rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    grid g { .x_start = p.nx, .x_end = p.nx + 1, .y_start = 1, .y_end = p.ny - 1 };
    return stencil(u_new, u_old, g, p);
}

// next_boundary function exchanges data with neighbor at rank + 1 and processes the rows that depend on the elements received.

double next_boundary(double* u_new, double* u_old, parameters p) {
    if (p.rank < p.nranks - 1) {
        // Receive bottom boundary from top rank
        MPI_Recv(u_old + (p.nx + 1) * p.ny, p.ny, MPI_DOUBLE, p.rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send top boundary to top rank, and
        MPI_Send(u_old + p.nx * p.ny, p.ny, MPI_DOUBLE, p.rank + 1, 1, MPI_COMM_WORLD);
    }
    grid g { .x_start = 1, .x_end = 2, .y_start = 1, .y_end = p.ny - 1 };
    return stencil(u_new, u_old, g, p);
}

