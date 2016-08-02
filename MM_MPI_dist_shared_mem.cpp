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
  // allocate a block for row-sized array [references] + actual content [rows*columns]
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
      worldRank,          // a task identifier
      amountWorkersWorld, // number of worker tasks
      mtype,              // message type
      rows,               // rows of matrix A sent to each worker
      averageRows,        // used to determine rows sent to each worker
      extraRows,          // used to determine additional rows sent to some workers  
      offset;

  vector<pair<int, int>> node_masters(0);

  MPI_Status status;

  MPI_Comm MPI_COMM_NODE; // communicator 
  int amountTasksNode, // size of shared memory
      sharedMemRank;   // rank in shared memory

  double** A;
  double** B;
  double** C;
  
  chrono::steady_clock::time_point startTime,
                                   endTime;

  srand (time(NULL));

  /*************************** MPI world ***************************/
  
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&worldRank);
  MPI_Comm_size(MPI_COMM_WORLD,&amountTasksWorld);
  amountWorkersWorld = amountTasksWorld - 1;
  
  if( amountWorkersWorld == 0 ) {
    cout << "This program needs more than one MPI thread!"<<endl;
    exit(-1);
  }
  
  /*************************** MPI node ***************************/
  
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &MPI_COMM_NODE);
  MPI_Comm_size(MPI_COMM_NODE, &amountTasksNode);
  MPI_Comm_rank(MPI_COMM_NODE, &sharedMemRank);

  /*************************** Find node masters ***************************/
  
  if (worldRank == MASTER) {
    mtype = FROM_MASTER;
    int nothing = 0;
    for (int dest=1; dest <= amountWorkersWorld; dest++) {
      MPI_Send(&nothing, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
    }
    mtype = FROM_WORKER;
    int isMaster;
    for (int j=1; j <= amountWorkersWorld; j++) {
      int source = j;
      MPI_Recv(&isMaster, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
      if(isMaster) {
        node_masters.push_back(make_pair(source, isMaster));
      }
    }
  } else {
    int comm = 0;
    if(sharedMemRank == 0) { // node master
      MPI_Recv(&comm, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      //send back amount of threads on node
      MPI_Send(&amountTasksNode, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    } else { // simple worker, send 0 back
      MPI_Recv(&comm, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
      MPI_Send(&comm, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
    }
  }
  
  /*************************** Master task ***************************/
  if (worldRank == MASTER) {
    A = allocDoubleArray(ROWS_A, COLUMNS_A);
    C = allocDoubleArray(ROWS_A, COLUMNS_B);
    if(DEBUG) cout<<"mpi_mm has started with "<< amountTasksWorld <<" tasks."<<endl;
    {/* Init */
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
   MPI_Barrier(MPI_COMM_WORLD);
  
  /*************************** Calculations ***************************/
  for(int iteration=0; iteration< ITER ; iteration++) {
    /*************************** Master task ***************************/
    if (worldRank == MASTER) {
      {/* Send matrix data to every node master */
        averageRows = ROWS_A / amountWorkersWorld;
        extraRows = ROWS_A % amountWorkersWorld;
        int offset = 0;
        mtype = FROM_MASTER;
        for (pair<int, int> p:node_masters) {
          int dest = p.first;
          int amountThreads = p.second;
          // send one more task per thread, when some tasks are left over
          int extraRowsSending = min(extraRows, amountThreads);
          rows = averageRows*amountThreads + extraRowsSending;
          extraRows -= extraRowsSending;
          
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
          offset += rows;
        }
      }
      {/* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (pair<int, int> p:node_masters) {
          int source = p.first;
          MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
          MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
          MPI_Recv(C[offset], rows*COLUMNS_B, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
          if(DEBUG) cout<<"[" << worldRank << "] Received results from task "<<source<<endl;
        }
      }
    } else {
      /*************************** Workers ***************************/
      MPI_Win nodeWin;
      double** smPointer;
      int nodeRows,
          nodeOffset;
      mtype = FROM_MASTER;
      
      MPI_Info win_info;
      MPI_Info_create(&win_info);
      MPI_Info_set(win_info, "alloc_shared_noncontig", "false");
      if(sharedMemRank == 0) {
        /*************************** Communication master ***************************/
        {/* allocating shared memory */
          MPI_Recv(&nodeOffset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
          if(DEBUG) cout<<"[" << worldRank << "] Received offset: "<<nodeOffset<<endl;
          MPI_Recv(&nodeRows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
          if(DEBUG) cout<<"[" << worldRank << "] Received rows: "<<nodeRows<<endl;
          int sizeAPart = nodeRows * sizeof(double)*(COLUMNS_A +1);
          int sizeBPart = ROWS_B * sizeof(double)*(COLUMNS_B +1);
          int sizeCPart = nodeRows * sizeof(double)*(COLUMNS_B +1);
          MPI_Win_allocate_shared(sizeAPart+sizeBPart+sizeCPart, sizeof(double), win_info, MPI_COMM_NODE, smPointer, &nodeWin);
          
          A = smPointer;
          B = smPointer + sizeAPart;
          C = smPointer + sizeAPart + sizeBPart;
        }
      } else {
        /*************************** Simple worker ***************************/
        MPI_Win_allocate_shared(0, 0, win_info, MPI_COMM_NODE, smPointer, &nodeWin);
      }
      MPI_Info_free(&win_info);
      // sync window
      MPI_Win_sync(nodeWin);
      MPI_Barrier(MPI_COMM_NODE);
      MPI_Win_sync(nodeWin);
      
      if(sharedMemRank == 0) {
        /*************************** Communication master ***************************/
        {// init arrays
          for(int i=0; i<nodeRows; i++) {
            A[i] = (double*)(A + nodeRows + i*COLUMNS_A);
            C[i] = (double*)(C + nodeRows + i*COLUMNS_B);
          }
          for(int i=0; i<ROWS_B; i++) {
            B[i] = (double*)(B + ROWS_B + i*COLUMNS_B);
          }
        }
        {// send rows + offsets to workers
          averageRows = nodeRows / amountWorkersWorld;
          extraRows = nodeRows % amountWorkersWorld;
          int offset = averageRows;
          
          for(int dest = 1; dest < amountTasksNode; dest++) {
            int rowsToWorker = (dest <= extraRows+1) ? averageRows+1 : averageRows;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_NODE);
            MPI_Send(&rowsToWorker, 1, MPI_INT, dest, mtype, MPI_COMM_NODE);
            offset+=rowsToWorker;
          }
          rows = averageRows;
        }
      } else {
        /*************************** Simple worker ***************************/
        // get shared memory access
        MPI_Aint smSize;
        int smDisp;
        
        double** smBasePtr;

        MPI_Win_shared_query(nodeWin, 0 /* communication master */, &smSize, &smDisp, &smBasePtr);
        // calculate pointer to A
        int amountRowsTotal = (smSize - ROWS_B*(COLUMNS_B+1)) / (COLUMNS_A + COLUMNS_B + 2); // 1. remove B, 2. A=rows + rows*cols & C=rows + rows*cols
        if(DEBUG) cout<<"calculated total amount of rows to be " << amountRowsTotal <<endl;
        int sizeA = amountRowsTotal*sizeof(double)*COLUMNS_A;
        int sizeB = ROWS_B*sizeof(double)*COLUMNS_B;
        A = smBasePtr;
        B = smBasePtr + sizeA;
        C = smPointer + sizeA + sizeB;

        // receive rows and offset
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_NODE, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_NODE, &status);
      }
      //calculate
      for (int row=0; row < rows; row++)
        for (int column=0; column < COLUMNS_A; column++)
          for(int j=0; j < COLUMNS_B; j++)
            C[row][j]+=  A[row][column] * B[column][j];
      // sync window
      MPI_Win_sync(nodeWin);
      MPI_Barrier(MPI_COMM_NODE);
      MPI_Win_sync(nodeWin);
      
      if (sharedMemRank == 0) {
        mtype = FROM_WORKER;
        MPI_Send(&nodeOffset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&nodeRows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(C[0], nodeRows*COLUMNS_B, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        free(A);
        free(B);
        free(C);
      }
      
    }
  }
  
  if (worldRank==0) {
    endTime = chrono::steady_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(endTime - startTime);
    double span = time_span.count();

    std::cout << "It took " << span << " seconds for " << ITER << " iterations, which means "<< span / ITER << " seconds on average."<<endl;
  }
  
  MPI_Finalize();
}
