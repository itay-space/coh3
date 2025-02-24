/*****************************************************/
/*         User defined OMPs                         */
/*****************************************************/

#include "omplib.h"

unsigned int OMPscratch(double e,int at,int zt,int ai,int zi,Optical *omp)
{
  unsigned int pf = OMPWilmoreHodgson(at,  ai, zi, omp);

   
  omp->v1  = 1; // 2m_redced/hbar^2 = 0.0474 MeV^-1 fm^-2
  omp->v2  = 0;  omp->v3  = 0 ;
  omp->ws1 =  0 ;  omp->ws2 = 0;  omp->ws3 =  0.0    ;
  omp->r0  =  1; // R = r0*A^(1/3)
  omp->rs  =  1;
  omp->a0  =  1 ;
  omp->vso1 = 0; // spin-orbit coupling 
  omp->avso = 1;
  omp->rvso = 1;

  return(pf);
}
