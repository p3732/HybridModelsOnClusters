#include <iostream>
#include <chrono>
#include <sstream>
#include "mpi.h"
#include <random>

#define MASTER       0  // task id of first task
#define FROM_MASTER  1  // setting a message type
#define FROM_WORKER  2  // setting a message type

#define DEBUG false

using namespace std;

double** allocDoubleArray(int rows, int columns) {
  // allocate an block for row-sized array [references] + actual content [rows*columns]
  double** array =(double **) malloc(rows * sizeof(double*) + rows*columns*sizeof(double));
  for(int i=0; i<rows; i++) {
    array[i] = (double*)(array + rows)+ i*columns;
  }

  return array;
}

/**
 * Needs argument list: rows in A, columns in A, columns in B, amount of iterations
 */
int main (int argc, char *argv[]) {

  /*************************** Parameters ***************************/
  int ITER = 1;
  int NRA  = 100;
  int NCA  = 100;
  int NCB  = 100;

  if(argc == 5) {
    {
      istringstream ss(argv[1]);
      int x = NRA;
      if (!(ss >> x))
        cerr << "Invalid number " << argv[1] << '\n';
      NRA = x;
    }
    {
      istringstream ss(argv[2]);
      int x = NCA;
      if (!(ss >> x))
        cerr << "Invalid number " << argv[1] << '\n';
      NCA=x;
    }
    {istringstream ss(argv[3]);
      int x=NCB;
      if (!(ss >> x))
        cerr << "Invalid number " << argv[1] << '\n';
      NCB=x;
    }
    {
      istringstream ss(argv[4]);
      int x=ITER;
      if (!(ss >> x))
        cerr << "Invalid number " << argv[2] << '\n';
      ITER=x;
    }
  } else {
    exit(-1);
  }
  int ROWS_A    = NRA;    // number of rows in matrix A
  int COLUMNS_A = NCA;    // number of columns in matrix A, equals number of rows in matrix B
  int ROWS_B    = NCA;    // number of rows in matrix B
  int COLUMNS_B = NCB;    // number of columns in matrix B

  int amountTasksWorld,   // number of tasks in partition
      rankWorld,          // a task identifier
      amountWorkersWorld, // number of worker tasks
      mtype,              // message type
      rows,               // rows of matrix A sent to each worker
      averageRows,        // used to determine rows sent to each worker
      extraRows,          // used to determine additional rows sent to some workers  
      offset;

  MPI_Status status;

  MPI_Comm sharedMemComm; // communicator 
  int amountTasksNode, // size of shared memory
      sharedMemRank;   // rank in shared memory

  double** A;
  double** B;
  double** C;
  
  B = allocDoubleArray(ROWS_B, COLUMNS_B);
  
  chrono::steady_clock::time_point startTime,
                                   endTime;

  srand (time(NULL));

  /*************************** MPI world ***************************/
  
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rankWorld);
  MPI_Comm_size(MPI_COMM_WORLD,&amountTasksWorld);
  amountWorkersWorld = amountTasksWorld - 1;
  
  if( amountWorkersWorld == 0 ) {
    cout << "This program needs more than one MPI thread!"<<endl;
    exit(-1);
  }
  
  /*************************** Master task ***************************/
  if (rankWorld == MASTER) {
    A = allocDoubleArray(ROWS_A, COLUMNS_A);
    C = allocDoubleArray(ROWS_A, COLUMNS_B);
    if(DEBUG) cout<<"mpi_mm has started with "<< amountTasksWorld <<" tasks."<<endl;
    {/* Init */
    // TODO move to workers ?
      if(DEBUG) cout<<"Initializing A array..."<<endl;
      for (int i=0; i<ROWS_A; i++)
        for (int j=0; j<COLUMNS_A; j++)
          A[i][j]= ((double) rand() / (RAND_MAX)) + 1;
      
      if(DEBUG) cout<<"Initializing B array..."<<endl;
      for (int i=0; i<ROWS_B; i++)
        for (int j=0; j<COLUMNS_B; j++)
          B[i][j]= ((double) rand() / (RAND_MAX)) + 1;
    }
    
    // Timer init
    startTime = chrono::steady_clock::now();
  }
  /*************************** Calculations ***************************/
  for(int i=0; i< ITER ; i++) {
    /*************************** Master task ***************************/
    if (rankWorld == MASTER) {
      {/* Send matrix data to every worker */
        averageRows = ROWS_A / amountWorkersWorld;
        extraRows = ROWS_A % amountWorkersWorld;
        int offset = 0;
        mtype = FROM_MASTER;
        for (int dest=1; dest <= amountWorkersWorld; dest++) {
          // send one more  task, when some tasks are left over
          rows = (dest <= extraRows) ? averageRows+1 : averageRows;
          if(DEBUG) cout<<"Sending "<<rows<<" rows to task "<<dest<<" offset="<< offset<<endl;
          // send offset
          MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
          if(DEBUG) cout<<"Sent offset "<<endl;
          // send amount of rows
          MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
          if(DEBUG) cout<<"Sent rows "<<endl;
          // send as many rows as calculated
          MPI_Send(A[offset], rows*COLUMNS_A, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
          if(DEBUG) cout<<"Sent A"<<endl;
          // send complete matrix B
          MPI_Send(B[0], ROWS_B*COLUMNS_B, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
          if(DEBUG) cout<<"Sent B"<<endl;
          offset = offset + rows;
        }
      }
      {/* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (int j=1; j<=amountWorkersWorld; j++) {
          int source = j;
          MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
          MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
          MPI_Recv(C[offset], rows*COLUMNS_B, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
          if(DEBUG) cout<<"Received results from task "<<source<<endl;
        }
      }
    } else {
      /*************************** Worker task ***************************/
      mtype = FROM_MASTER;
      
      //store offset and amount of rows, to send back later
      MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      if(DEBUG) cout<<"Received offset: "<<offset<<endl;
      MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      if(DEBUG) cout<<"Received rows: "<<rows<<endl;
      
      A = allocDoubleArray(rows, COLUMNS_A);
      C = allocDoubleArray(rows, COLUMNS_B);
      
      MPI_Recv(A[0], rows*COLUMNS_A, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      if(DEBUG) cout<<"Received A"<<endl;
      MPI_Recv(B[0], ROWS_B*COLUMNS_B, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      if(DEBUG) cout<<"Received B"<<endl;
      if(DEBUG) cout<<"received all at "<< rankWorld << endl;

      /*** Do matrix multiply sharing iterations on outer loop ***/
      #pragma omp parallel for shared(A,B,C) schedule(static)
      for (int row=0; row < rows; row++)
        for (int column=0; column < COLUMNS_A; column++)
          for(int j=0; j < COLUMNS_B; j++)
            C[row][j]+=  A[row][column] * B[column][j];

      mtype = FROM_WORKER;
      MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(C[0], rows*COLUMNS_B, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      free(A);
      free(C);
    }
    if(DEBUG && rankWorld==MASTER) {
      for(int j=0; j<ROWS_A; j++) {
        for(int k=0; k<COLUMNS_B; k++) {
          cout << C[j][k]<<" ";
        }
        cout <<endl;
      }
    }
  }
  
  if (rankWorld==0) {
    endTime = chrono::steady_clock::now();
    chrono::duration<double> time_span =
        chrono::duration_cast<chrono::duration<double>>(endTime - startTime);
    double span = time_span.count();

    std::cout << "It took " << span << " seconds for " << ITER << " iterations, which means "
              << span / ITER << " seconds on average."<<endl;
  }
  
  MPI_Finalize();
}
