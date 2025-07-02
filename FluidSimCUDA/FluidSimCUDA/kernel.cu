//CPU Code taken from: https://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf
//Credit to Jos Stam

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <windows.h>
#include <string>
#include <array>
#include <ctime>
#include <iomanip>

#define ITERATIONS 20
#define IX(i,j) ((i) + (N+2)*(j)) //index
#define SWAP(x0,x) {double* tmp = x0; x0 = x; x = tmp;} //swap 2 pointers
#define LERP(val1,val2,x) (val1 + x*(val2-val1)) //linear interpolate macro used in advect

/* Creates and initializes the passed GLFWwindow object, also calls glewInit() and makes sure it is GLEW_OK */
void initializeWindow(GLFWwindow*& window, int width, int height, const char* name) {
    if (!glfwInit()) {
        std::cout << "glfwInit() didn't work" << std::endl;
        return;
    }

    window = glfwCreateWindow(width, height, name, NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        std::cout << "passed GLFWwindow* reference doesn't exist" << std::endl;
        return;
    }

    glfwMakeContextCurrent(window);

    /* initialize glew, so we can use all the other openGL functions that were introduced past version 1.1 */
    if (glewInit() != GLEW_OK) {
        std::cout << "glewInit != GLEW_OK" << std::endl;
        return;
    }
}

/* Goes through inputArray of certain values, i.e. velocities, densities, and puts those in the given valueArray, boundary cells included*/
__global__ void addSources(double *x, double *sourceArray, int N, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        x[IX(i - 1, j)] += dt * sourceArray[IX(i - 1, j)];
    } else if (blockIdx.y == 0 && threadIdx.y == 0) {
        x[IX(i, j - 1)] += dt * sourceArray[IX(i, j - 1)];
    } else if (blockIdx.x == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
        x[IX(i + 1, j)] += dt * sourceArray[IX(i + 1, j)];
    } else if (blockIdx.y == gridDim.y - 1 && threadIdx.y == blockDim.y - 1) {
        x[IX(i, j + 1)] += dt * sourceArray[IX(i, j + 1)];
    }
    x[IX(i, j)] += dt * sourceArray[IX(i, j)];
}

__global__ void set_bnd_rims(int N, int b, double* x) {
    int i = threadIdx.x+1;
    x[IX(0, i)] = (b == 1) ? -x[IX(1, i)] : x[IX(1, i)];
    x[IX(N + 1, i)] = (b == 1) ? -x[IX(N, i)] : x[IX(N, i)];
    x[IX(i, 0)] = (b == 2) ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, N + 1)] = (b == 2) ? -x[IX(i, N)] : x[IX(i, N)];
}

/* Sets the boundaries. Free-slip condition */
void set_bnd(int N, int b, double* x) {
    set_bnd_rims <<<1, N>>> (N, b, x);
    cudaDeviceSynchronize();
    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
    x[IX(0, N + 1)] = 0.5 * (x[IX(0, N)] + x[IX(1, N + 1)]);
    x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

//Taken from: https://www.cse.chalmers.se/edu/year/2018/course/TDA361/Advanced%20Computer%20Graphics/GpuGems-FluidDynamics.pdf
__global__ void jacobiStep(double *x, double *b, double alpha, double beta, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    x[IX(i, j)] = (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)] + alpha * b[IX(i, j)]) / beta;
}

/* Handles diffusion. In one time interval the passed quantity should gradually become the same as its 4 adjacent cells. */
void diffuse(double* x, double* x0, double diff, int N, int b, double dt, dim3 dimBlock, dim3 dimGrid) {
    double alpha = ((1.0 / N) * (1.0 / N)) / (diff * dt);

    for (int k = 0; k < ITERATIONS; k++) {
        jacobiStep <<<dimGrid, dimBlock>>> (x, x0, alpha, 4 + alpha, N);
    }
    cudaDeviceSynchronize();

    /* When we're diffusing densities, this line is irrelevant, however when diffusing velocities it is crucial.
       This line makes sure that whenever we update a cell, that is next to a boundary cell's velocity,
       we immediately reflect that cell's velocity on the boundary cell, so that the boundary cell can now,
       in the next jacobi iteration, steer the velocity of the cell away from the boundary, so the cell points more in the
       boundaries tangential direction. */
    set_bnd(N, b, x);
}

/* Moves the fluid along its vectors (roughly), using semi-lagranian advection */
__global__ void advect(double* x, double* x0, double* u, double* v, int N, int b, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    double particleX = 0, particleY = 0, particleX_INT = 0, particleY_INT = 0, particleX_FRACT = 0, particleY_FRACT = 0, lerpX1 = 0, lerpX2 = 0, interpolatedValue = 0;

    /* Determine "particle's" position */
    particleX = i - u[IX(i, j)] * dt * N;
    particleY = j - v[IX(i, j)] * dt * N;
    /* Handles overflow by capping the max value of particleX and particleY at the respective row/columns boundary cell.
       This way no matter how big the velocity at a certain point is, particleX will always be in the same row and particleY
       will never be out of bounds. */
    if (particleX < 0.5) particleX = 0.5; if (particleX > N + 0.5) particleX = N + 0.5; //handle x-overflow
    if (particleY < 0.5) particleY = 0.5; if (particleY > N + 0.5) particleY = N + 0.5; //handle y-overflow
    particleX_FRACT = modf(particleX, &particleX_INT);
    particleY_FRACT = modf(particleY, &particleY_INT);
    particleY_INT++;
    
    // Interpolate the values of 4 cells to approximate the particle's value
    lerpX1 = LERP(x0[IX(int(particleX_INT), int(particleY_INT))], x0[IX(int(particleX_INT + 1), int(particleY_INT))], particleX_FRACT);
    lerpX2 = LERP(x0[IX(int(particleX_INT), int(particleY_INT - 1))], x0[IX(int(particleX_INT + 1), int(particleY_INT) - 1)], particleX_FRACT);
    x[IX(i, j)] = LERP(lerpX2, lerpX1, particleY_FRACT);
}

__global__ void computeDivergence(double *div, double *u, double *v, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    div[IX(i, j)] = 0.5 * N * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]); //since N = 1/dx
}

__global__ void subtractGradient(double *p, double *u, double *v, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N;
    v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N;
}

/* Clears any divergence and thus forces incompressibility */
void project(int N, double* u, double* v, double* p, double* div, dim3 dimBlock, dim3 dimGrid) {
    cudaMemset(p, 0, (N + 2) * (N + 2) * sizeof(double));

    /* Compute divergence of u and v */
    computeDivergence <<<dimGrid, dimBlock>>> (div, u, v, N);
    cudaDeviceSynchronize();

    /* Make divergence consistent along boundaries, i.e. just copy over the values of the already present values to the boundary cells
       and also initialize all boundary cells in p to 0 */
    set_bnd(N, 0, div); set_bnd(N, 0, p);

    for (int k = 0; k < ITERATIONS; k++) {
        jacobiStep <<<dimGrid, dimBlock>>> (p, div, -1.0/(N*N), 4.0, N);
    }
    cudaDeviceSynchronize();

    /* as approximation let the boundary cells be the same value as the cells that surround them, better than just having them all be 0
       more consistency and also more expected behaviour */
    set_bnd(N, 0, p);

    /* Now calculate the gradient (i.e. del*p) of our newly determined scalar field p and subtract it */
    subtractGradient <<<dimGrid, dimBlock>>> (p, u, v, N);
    cudaDeviceSynchronize();

    /* vector field changed thus update boundary values accordingly */
    set_bnd(N, 1, u); set_bnd(N, 2, v);
}

/* Advanced the velocites by one timestep */
void velocity_step(int N, double dt, double diff, double* u, double* u_prev, double* v, double* v_prev, dim3 dimBlock, dim3 dimGrid) {
    addSources <<<dimGrid, dimBlock>>> (u, u_prev, N, dt);
    cudaDeviceSynchronize();
    addSources <<<dimGrid, dimBlock>>> (v, v_prev, N, dt);
    cudaDeviceSynchronize();
    SWAP(u_prev, u); 
    diffuse(u, u_prev, diff, N, 1, dt, dimBlock, dimGrid);
    SWAP(v_prev, v); 
    diffuse(v, v_prev, diff, N, 2, dt, dimBlock, dimGrid);
    project(N, u, v, u_prev, v_prev, dimBlock, dimGrid);
    SWAP(u_prev, u); 
    SWAP(v_prev, v);
    advect <<<dimGrid, dimBlock>>> (u, u_prev, u_prev, v_prev, N, 1, dt);
    cudaDeviceSynchronize();
    set_bnd(N, 1, u);
    advect <<<dimGrid, dimBlock>>> (v, v_prev, u_prev, v_prev, N, 2, dt);
    cudaDeviceSynchronize();
    set_bnd(N, 2, v);
    project(N, u, v, u_prev, v_prev, dimBlock, dimGrid);
}

/* Advances the densities by one timestep */
void density_step(int N, double dt, double diff, double* dens, double* dens_prev, double* u, double* v, dim3 dimBlock, dim3 dimGrid) {
    addSources <<<dimGrid, dimBlock>>> (dens, dens_prev, N, dt);
    cudaDeviceSynchronize();
    SWAP(dens_prev, dens);
    diffuse(dens, dens_prev, diff, N, 0, dt, dimBlock, dimGrid);
    SWAP(dens_prev, dens);
    advect <<<dimGrid, dimBlock>>> (dens, dens_prev, u, v, N, 0, dt);
    cudaDeviceSynchronize();
}

//if you want to parallelize this look at this: https://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf
/* Draws the grid based on the values in the passed 'densityArray'. The densityArray includes boundary cells. It is up to the caller to adjust the array to their need */
void drawGrid(double* densityArray, int N, double gridSize) {
    int arrayIndex;
    double gridSizeTimes2 = gridSize * 2;
    double x = -1.0;
    double y = -1.0;

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    glBegin(GL_QUADS);
    for (int j = 1; j < N + 1; j++) {
        for (int i = 1; i < N + 1; i++) {
            arrayIndex = IX(i, j);
            glColor3d(densityArray[arrayIndex], densityArray[arrayIndex], densityArray[arrayIndex]);
            glVertex2d(x, y);
            glVertex2d(x + gridSizeTimes2, y);
            glVertex2d(x + gridSizeTimes2, y + gridSizeTimes2);
            glVertex2d(x, y + gridSizeTimes2);
            x += gridSizeTimes2;
        }
        x = -1.0;
        y += gridSizeTimes2;
    }
    glEnd();
}

/* The main simulation loop. Call this to execute the 2D eulerian fluid simulation. Function returns when passed GLFWwindow* object is closed. */
void simulate(int N, int width, double gridSize, double dt, double* dens, double* dens_prev, double* u, double* u_prev, double* v, double* v_prev, double diff, double dyeIntensity, GLFWwindow* window, dim3 dimBlock, dim3 dimGrid, bool benchmark) {
    cudaSetDevice(0);

    int frames = 0;
    double avg[30] = {0};
    int it = 0;
    time_t start = time(0);
    time_t prev = time(0);
    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window)) {
        if (benchmark && time(0) - prev >= 1) {
            printf("time(0)-start: %f\n", difftime(time(0), start));
            printf("time(0)-prev: %f\n", difftime(time(0), prev));
            printf("Cells/s: %d\n", frames * (N*N));
            avg[it++] = (frames*(N*N)) / (difftime(time(0), prev));
            printf("Avg[%d]: %f\n", it-1, avg[it-1]);
            frames = 0;
            prev = time(0);
        }
        if (benchmark && time(0) - start >= 30) break;
        glClear(GL_COLOR_BUFFER_BIT);

        //Add sources here ---------------------------------------------------------
        //i = x-coordinate, j = y-coordinate, u_prev = x-velocity, v_prev = y-velocity
        //make sure they are within (1,1) and (N,N) since there is an additional boundary layer
        int i = 200;
        int j = 255;
        u_prev[IX(i, j)] = 0.5;
        dens_prev[IX(i, j)] = 5;
        v_prev[IX(i, j)] = 0;

        i = 300;
        j = 255;
        u_prev[IX(i, j)] = -0.5;
        dens_prev[IX(i, j)] = 5;
        v_prev[IX(i, j)] = 0;
        //--------------------------------------------------------------------------

        velocity_step(N, dt, diff, u, u_prev, v, v_prev, dimBlock, dimGrid);
        density_step(N, dt, diff, dens, dens_prev, u, v, dimBlock, dimGrid);
        drawGrid(dens, N, gridSize);

        cudaMemset(u_prev, 0, (N + 2) * (N + 2) * sizeof(double));
        cudaMemset(v_prev, 0, (N + 2) * (N + 2) * sizeof(double));
        cudaMemset(dens_prev, 0, (N + 2) * (N + 2) * sizeof(double));

        glfwSwapBuffers(window);
        glfwPollEvents();
        frames++;
    }

    time_t end = time(0);
    double total_avg = 0;
    for (int i = 0; i < it; i++) {
        total_avg += avg[i];
    }
    printf("Avg Cells/s: %f\nTotal_Avg: %f\nSeconds: %f\n", total_avg/(double)it, total_avg, difftime(end, start));
    fflush(stdout);
}

int main() {
    GLFWwindow* window;
    int width = 1440;
    int height = 1440;

    //Change N here (N must be a multiple of 32!)-----------------------------
    const int N = 12288;
    //------------------------------------------------------------------------

    dim3 dimGrid(N/32, N/32);
    dim3 dimBlock(32, 32);
    const double dt = 0.5;
    double gridSize = 1.0 / N;
    const int totalAmountOfCells = (N + 2) * (N + 2);
    double *u = nullptr, *u_prev = nullptr, *v = nullptr, *v_prev = nullptr, *dens = nullptr, *dens_prev = nullptr, *p = nullptr, *div = nullptr;

    //choose which GPU to run on (in this case there's only 1)
    cudaSetDevice(0);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed. Do you have a CUDA compatible GPU in your system?\n");
    }

    cudaMallocManaged(&u, totalAmountOfCells * sizeof(double));
    cudaMallocManaged(&u_prev, totalAmountOfCells * sizeof(double));
    cudaMallocManaged(&v, totalAmountOfCells * sizeof(double));
    cudaMallocManaged(&v_prev, totalAmountOfCells * sizeof(double));
    cudaMallocManaged(&dens, totalAmountOfCells * sizeof(double));
    cudaMallocManaged(&dens_prev, totalAmountOfCells * sizeof(double));

    initializeWindow(window, width, height, "ouroborous 41-100%");
    simulate(N, width, gridSize, dt, dens, dens_prev, u, u_prev, v, v_prev, 0.000000002299, 6, window, dimBlock, dimGrid, true);
    glfwTerminate();

    cudaFree(u);
    cudaFree(u_prev);
    cudaFree(v);
    cudaFree(v_prev);
    cudaFree(dens);
    cudaFree(dens_prev);

    //needed for profiling with NSight
    cudaDeviceReset();
    return 0;
}
