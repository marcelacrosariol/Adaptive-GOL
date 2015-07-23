#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

#define RANDVAL 1984
#define BLOCK_SIZE 16
#define DIM     4 // Linear dimension of our grid - not counting ghost cells

// Create an array that stores the number of rows of the subGrid in each device
__host__ void RowCount(int devCount,int *subGridSize){
    for (int i = 0; i < devCount; i++){
        if (DIM % devCount == 0)
            subGridSize[i] = DIM / devCount;
        else{
            if (i == 0) // If it is not possible to divide the rows equally between the devices, the first GPU will receive more rows than the others
                subGridSize[i] = ((int) DIM / devCount) + DIM % devCount;
            else
                subGridSize[i] = (int) DIM / devCount;
        }
    }
}

// Return the number of rows that exist in the main grid before the first row of the actual subgrid
__host__ int rowsBefore(int device, int *subGridSize){
    if (device == 0)
        return 0;
    else if (device ==1)
        return subGridSize[0];
    else
        return subGridSize[device-1] + rowsBefore(device-1, subGridSize);
}

__host__ int getLastRow(int device, int *subGridSize){
    if (device == 0)
        return subGridSize[0];
    else
        return subGridSize[device] + getLastRow(device-1, subGridSize);
}

__host__ void buildHaloCells(int *h_grid){
    // Copy halo rows
    for (int j = 1; j<= DIM; j++){

        // Copy first real row to last halo row
        h_grid[(DIM+1) * (DIM+2) + j] = h_grid[1 * (DIM+2) + j];

        // Copy last real row to first halo row
        h_grid[j] = h_grid[DIM * (DIM+2) + j];
    }

    // Copy halo columns
    for (int i = 0; i<= DIM+1; i++){

        // Copy first real column to last halo column
        h_grid[i * (DIM+2) + (DIM+1)] = h_grid[i* (DIM+2) + 1];

        // Copy last real column to first halo column
        h_grid[i * (DIM+2)] = h_grid[i * (DIM+2) + DIM];
    }
}

__host__ void buildSubGrid(int *h_grid, int *h_subGrid, int firstRow, int lastRow, int d){
    for(int i = firstRow-1; i<= lastRow+1; i++){
        for(int j = 0; j<= DIM+1; j++){
            h_subGrid[i * (DIM+2) + j] = h_grid[i * (DIM+2) + j];
        }
    }
}


__global__ void ghostRows(int *grid){
    // We want id ∈ [1,DIM]
    int id = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (id <= DIM){
        //Copy first real row to bottom ghost row
        grid[(DIM+2)*(DIM+1)+id] = grid[(DIM+2)+id];

        //Copy last real row to top ghost row
        grid[id] = grid[(DIM+2)*DIM + id];
    }
}

__global__ void ghostCols(int *grid){
    // We want id ∈ [0,DIM+1]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id <= DIM+1){
        //Copy first real column to right most ghost column
        grid[id*(DIM+2)+DIM+1] = grid[id*(DIM+2)+1];

        //Copy last real column to left most ghost column
        grid[id*(DIM+2)] = grid[id*(DIM+2) + DIM];
    }
}

__global__ void GOL(int *grid, int *newGrid, int firstRow){
    // We want id ∈ [1,DIM]
    int iy = blockDim.y * blockIdx.y + threadIdx.y  + firstRow;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int id = iy * (DIM+2) + ix;


    int numNeighbors;

    if (iy <= DIM && ix <= DIM) {

        // Get the number of neighbors for a given grid point
        numNeighbors = grid[id+(DIM+2)] + grid[id-(DIM+2)] //upper lower
        + grid[id+1] + grid[id-1]             //right left
        + grid[id+(DIM+3)] + grid[id-(DIM+3)] //diagonals
        + grid[id-(DIM+1)] + grid[id+(DIM+1)];

        int cell = grid[id];

        //printf("firstrow: %d ID: %d Grid[%d]: %d cell: %d Neighboors: %d \n", firstRow, id, id, grid[id], cell ,numNeighbors);

        // Here we have explicitly all of the game rules
        if (cell == 1 && numNeighbors < 2)
            newGrid[id] = 0;
        else if (cell == 1 && (numNeighbors == 2 || numNeighbors == 3))
            newGrid[id] = 1;
        else if (cell == 1 && numNeighbors > 3)
            newGrid[id] = 0;
        else if (cell == 0 && numNeighbors == 3)
            newGrid[id] = 1;
        else
            newGrid[id] = cell;
    }
}


int main(int argc, char* argv[]){

    int devCount;
    cudaGetDeviceCount(&devCount); // Get the number of devices that the system have
    printf("There are %d devices \n", devCount);
    // If there is no GPU, it is not possible to run this version of Game of Life
    if (devCount == 0){
        printf("There are no devices in this machine!");
        return 0; // if there is no GPU, then break the code
    }
    
    int i, j, iter;
    int alive = 0, lim = DIM;
    int *h_grid;
    size_t gridBytes;
    
    gridBytes = sizeof(int)*(DIM+2)*(DIM+2); // 2 added for periodic boundary condition ghost cells
    // Alocate memory for host grid
    h_grid = (int*)malloc(gridBytes);
    
    srand(RANDVAL);
    // Assign random value to cells of the grid
    #pragma omp parallel for private(i,j)
    for(i = 1; i<=DIM; i++) {
        for(j = 1; j<=DIM; j++) {
           h_grid[i*(DIM+2)+j] = rand() % 2;
        }
    } // End of pragma


printf("\n");
for(int i = 1; i <= DIM; i++){
for(int j = 1; j <= DIM; j++){
printf("%d  ", h_grid[i*(DIM+2)+j]);
}
printf("\n");
}


    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    int  linGrid = (int)ceil(DIM/(float)BLOCK_SIZE);
    dim3 gridSize(linGrid,linGrid,1);

    dim3 cpyBlockSize(BLOCK_SIZE,1,1);
    dim3 cpyGridRowsGridSize((int)ceil(DIM/(float)cpyBlockSize.x),1,1);
    dim3 cpyGridColsGridSize((int)ceil((DIM+2)/(float)cpyBlockSize.x),1,1);
    
    if (devCount == 1){
        int *d_grid, *d_newGrid, *d_tmpGrid;
        
        // Allocate device grids - if there is more than 1 thread, It'll allocate memory in each device
        cudaMalloc(&d_grid, gridBytes);
        cudaMalloc(&d_newGrid, gridBytes);
        
        // Copy over initial game grid (Dim-1 threads)
        cudaMemcpy(d_grid, h_grid, gridBytes, cudaMemcpyHostToDevice);

        for (iter = 0; iter < lim; iter ++){

            ghostRows<<<cpyGridRowsGridSize, cpyBlockSize>>>(d_grid);
            ghostCols<<<cpyGridColsGridSize, cpyBlockSize>>>(d_grid);
            GOL<<<gridSize, blockSize>>>(d_grid, d_newGrid,1);

            // Swap our grids and iterate again
            d_tmpGrid = d_grid;
            d_grid = d_newGrid;
            d_newGrid = d_tmpGrid;
        }

        // Copy back results and sum
        cudaMemcpy(h_grid, d_grid, gridBytes, cudaMemcpyDeviceToHost);
        
        // calculate the total of cells alive after the iteractions
        //#pragma omp parallel for private(i,j,alive)
        for (i = 1; i <= DIM; i++){
            for ( j =1 ; j <= DIM; j++){
                alive += h_grid[i*(DIM+2)+j];
            }
        }  // end of prama
        
        printf("There are %d cells alive after the last iteration\n", alive);
        
        // Release memory
        cudaFree(d_grid);
        cudaFree(d_newGrid);
        free(h_grid);
        
        return 1;
    }
    
    if (devCount > 1){

        int *h_SubGridSize;
        int CPUthreadId, currentDevice, firstRow, lastRow;
        int *h_subGrid;
        int *d_subGrid, *d_tempSub;
        size_t subBytes;

        h_SubGridSize = (int*)malloc(sizeof(int)*devCount); // Allocate memory for the subGridSize, which stores the number of elements in each subGrids
        RowCount(devCount, h_SubGridSize); // Calculate the size of the subgrid in each GPU

        buildHaloCells(h_grid);

        printf("\n");
        for(int i = 0; i <= DIM+1; i++){
            for(int j = 0; j <= DIM+1; j++){
                printf("%d  ", h_grid[i*(DIM+2)+j]);
            }
        printf("\n");
        }

        omp_set_num_threads(devCount);

        for (iter = 0; iter < lim; iter ++){

            #pragma omp parallel
            {
                #pragma omp ordered for private(currentDevice, CPUthreadId, h_subGrid, subBytes, firstRow, lastRow)
                {
                for (int device = 0; device < devCount; device++){
                CPUthreadId = omp_get_thread_num(); // Get the id of the actual thread

                currentDevice = CPUthreadId;
                cudaSetDevice(device); // Set device to be used

                subBytes = sizeof(int)* (DIM+2) * (DIM+2);  //(h_SubGridSize[currentDevice]+2) * (DIM+2);  number of rows + 2 halo/ghost rows + 2 halo/ghost columns
                h_subGrid = (int*)malloc(subBytes); //allocate memory for the subGrid

                // Allocate device grids - if there is more than 1 thread, It'll allocate memory in each device
                cudaMalloc(&d_subGrid, subBytes);
                cudaMalloc(&d_tempSub, subBytes);

                // Calculates the first row of the submatrix in the main matrix  - Does not count the ghost rows
                firstRow = rowsBefore(currentDevice, h_SubGridSize) + 1;
                // Calculates the last row of the submatrix in the main matrix
                lastRow = getLastRow(currentDevice, h_SubGridSize);

                buildSubGrid(h_grid, h_subGrid, firstRow, lastRow, currentDevice);

                    printf("iter : %d \n", iter);
                    printf("\n printing subgrid gpu %d  \n", currentDevice);
                    printf("firstrow: %d lastRow:%d \n", firstRow, lastRow);
                    for(int i = firstRow-1; i <= lastRow+1; i++){
                        for(int j = 0; j <= DIM+1; j++){
                            printf("%d  ", h_subGrid[i*(DIM+2)+j]);
                        }
                        printf("\n");
                    }


                cudaMemcpy(d_subGrid, h_subGrid, subBytes, cudaMemcpyHostToDevice);

                // call GOL function and the new values will go to the d_tempSub grid
                GOL<<<gridSize, blockSize>>>(d_subGrid, d_tempSub, firstRow);

                free(h_subGrid);
                h_subGrid = (int*)malloc(subBytes); //allocate memory for the subGrid

                cudaMemcpy(h_subGrid, d_tempSub, subBytes, cudaMemcpyDeviceToHost);

               for(int i = firstRow; i <= lastRow; i++){
                    for(int j = 1; j <= DIM; j++){
                        h_grid[i*(DIM+2)+j] = h_subGrid[i*(DIM+2)+j];
                    }
                }
}}} // End pragma
        } // End iteration


        printf("\n___\n\n");
        for(int i = 1; i<=DIM; i++) {
            for(int j = 1; j<=DIM; j++) {
                printf("%d  ", h_grid[i*(DIM+2)+j]);
            }
            printf("\n");
        }


        for (int i = 1; i <= DIM; i++){
            for (int j =1 ; j <= DIM; j++){
                alive += h_grid[i*(DIM+2)+j];
            }
        }  // end of prama

        printf("There are %d cells alive after the last iteration\n", alive);

        // Release memory
        free(h_grid);
        cudaFree(d_tempSub);

        return 1;
    }
}
    