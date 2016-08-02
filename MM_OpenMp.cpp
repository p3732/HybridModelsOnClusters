#include <iostream>
#include <chrono>
#include <sstream>
#include <omp.h>
#include <random>

#define MASTER      0  // task id of first task
#define FROM_MASTER 1  // setting a message type
#define FROM_WORKER 2  // setting a message type

#define DEBUG false

using namespace std;

double** allocDoubleMatrix(int rows, int columns) {
  // allocate an block for row-sized array [references] + actual content [rows*columns]
  double** array =(double **) malloc(rows * sizeof(double*) + rows*columns*sizeof(double));
  for(int i=0; i<rows; i++) {
    array[i] = (double*)(array + rows)+ i*columns;
  }

  return array;
}

int main (int argc, char *argv[]) {
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

  double** A = allocDoubleMatrix(ROWS_A, COLUMNS_A);
  double** B = allocDoubleMatrix(ROWS_B, COLUMNS_B);
  double** C = allocDoubleMatrix(ROWS_A, COLUMNS_B);

  chrono::steady_clock::time_point startTime,
                                   endTime;
  srand (time(NULL));
  // Timer init
  startTime = chrono::steady_clock::now();
  /* Init */
  if(DEBUG) cout<<"Initializing arrays..."<<endl;
  #pragma omp parallel for
  for (int i=0; i<ROWS_A; i++)
    for (int j=0; j<COLUMNS_A; j++)
      A[i][j]= ((double) rand() / (RAND_MAX)) + 1;
  #pragma omp parallel for
  for (int i=0; i<COLUMNS_A; i++)
    for (int j=0; j<COLUMNS_B; j++)
      B[i][j]= ((double) rand() / (RAND_MAX)) + 1;

  for(int iteration=0; iteration< ITER ; iteration++) {

/*      if (DEBUG){
        int tid = omp_get_thread_num();
        if (tid == 0)
          cout<<"OpenMP matrix multiplication with " << omp_get_num_threads() << " threads.\n"<< endl;
        // cout<<"Thread "<<tid<<" starting\n"<<endl;
      }
  */  
    /*** Do matrix multiply sharing iterations on outer loop ***/
    #pragma omp parallel for
    for (int i=0; i < ROWS_A; i++)
      for (int k=0; k < COLUMNS_A; k++)
        for(int j=0; j < COLUMNS_B; j++)
          C[i][j]+=  A[i][k] * B[k][j];
    if(DEBUG) {
      for(int i=0; i<ROWS_A; i++) {
        for(int j=0; j<COLUMNS_B; j++) {
          cout << C[i][j]<<" ";
        }
        cout <<endl;
      }
    }
  }
  {
    endTime = chrono::steady_clock::now();
    chrono::duration<double> time_span =
        chrono::duration_cast<chrono::duration<double>>(endTime - startTime);
    double span = time_span.count();

    std::cout << "It took " << span << " seconds for " << ITER << " iterations, which means "
              << span / ITER << " seconds on average."<<endl;
  }
}
