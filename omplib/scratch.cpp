/*****************************************************/
/*         User defined OMPs                         */
/*****************************************************/

#include "omplib.h"

unsigned int OMPscratch(double e,int at,int zt,int ai,int zi,Optical *omp)
{
  unsigned int pf = OMPWilmoreHodgson(at,  ai, zi, omp);

   
   omp->v1  = 10 ;  omp->v2  = 0;  omp->v3  = 0 ;
   omp->ws1 =  0 ;  omp->ws2 = 0;  omp->ws3 =  0.0    ;
   omp->r0  =  1 / pow(at,1.0/3.0); // R = r0*A^(1/3) = 1 
   omp->rs  =  1;
   omp->a0  =  1 ;
   omp->as  =  1 ;

  return(pf);
}
