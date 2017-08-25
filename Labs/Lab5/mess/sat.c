/*
 *   Circuit Satisfiability, Version 1
 *
 *   This MPI program determines whether a circuit is
 *   satisfiable, that is, whether there is a combination of
 *   inputs that causes the output of the circuit to be 1.
 *   The particular circuit being tested is "wired" into the
 *   logic of function 'check_circuit'. All combinations of
 *   inputs that satisfy the circuit are printed.
 *
 *   Programmed by Michael J. Quinn
 *   Last modification: 3 September 2002
 *
 *   Modified by Sanjay Rajopadhye (Sept 11 2008)
 *   Extended to 20 bit function that has 18 distinct outputs
 *
*/

#include "mpi.h"
#include <stdio.h>
#define MAXARG 1048576

int main (int argc, char *argv[]) {
   int i;
   int id;               /* Process rank */
   int p;                /* Number of processes */
   int solutions, global_solutions;
   double elapsed_time;
   int check_circuit (int, int);
   
   MPI_Init (&argc, &argv);
   MPI_Barrier(MPI_COMM_WORLD);
   elapsed_time = - MPI_Wtime();
   MPI_Comm_rank (MPI_COMM_WORLD, &id);
   MPI_Comm_size (MPI_COMM_WORLD, &p);

   solutions = 0;
   for (i = id; i < MAXARG; i += p)
      solutions += check_circuit (id, i);

   MPI_Reduce(&solutions, &global_solutions, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   if (id == 0) printf ("Process 0: %d solutions\n", global_solutions);
   printf ("Process %d is done\n", id);
   fflush (stdout);
   

   elapsed_time += MPI_Wtime();
   if (id == 0) printf ("Total Execution Time is %f \n", elapsed_time);
   MPI_Finalize();
   return 0;
}

/* Return 1 if 'i'th bit of 'n' is 1; 0 otherwise */
#define EXTRACT_BIT(n,i) ((n&(1<<i))?1:0)

int check_circuit (int id, int z) {
   int v[20];        /* Each element is a bit of z */
   int i;

   for (i = 0; i < 20; i++) v[i] = EXTRACT_BIT(z,i);
   if ((v[0] || v[1]) && (!v[1] || !v[3]) && (v[2] || v[3])
      && (!v[3] || !v[4]) && (v[4] || !v[5])
      && (v[5] || !v[6]) && (v[5] || v[6])
      && (v[6] || !v[15]) && (v[7] || !v[8])
      && (!v[7] || !v[13]) && (v[8] || v[9])
      && (v[8] || !v[9]) && (!v[9] || !v[10])
      && (v[9] || v[11]) && (v[10] || v[11])
      && (v[12] || v[13]) && (v[13] || !v[14])
      && (v[14] || v[15])
      && ((v[16] && v[17] && v[18] && v[19]) || 
	  (v[16] && v[17] && v[18] && !v[19]))
       )
     {
       printf ("%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", id,
         v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],
         v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19]);
	 fflush (stdout);
       return 1;
     }
   else
     return 0;
}
