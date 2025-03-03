/******************************************************************************/
/*  setupinit.cpp                                                             */
/*        setting up reaction chain                                           */
/******************************************************************************/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

#include "physicalconstant.h"
#include "structur.h"
#include "coh.h"
#include "setupinit.h"
#include "nucleus.h"
#include "masstable.h"
#include "terminate.h"
#include "mt19937.h"

static void setupInitMT (void);
static bool firstcall = true;

// user-defined uncommon incident particle
static const int    userdefZ    = 0;
static const int    userdefA    = 4;
static const int    userdefJ2   = 0;
static const double userdefMass = 4.0;
static const double userdefMassExcess = 4*ENEUTRON - 0.42;
//static const double userdefMassExcess = 4*ENEUTRON - 1.0;

// static const int    userdefZ    = 0;
// static const int    userdefA    = 6;
// static const int    userdefJ2   = 0;
// static const double userdefMass = 6.0;
// static const double userdefMassExcess = 6*ENEUTRON - 1.0;


/**********************************************************/
/*     Initialize All Data                                */
/**********************************************************/
void setupInitSystem(const int mc, Pdata *pdt)
{
  /*** store particle mass, spin, and identifiers */
  for(int p=0 ; p<mc ; p++){
    pdt[p].pid = unknown;
    pdt[p].omp = 0;
    switch(p){
    case neutron:
      pdt[p].za.setZA(0,1);    pdt[p].pid = neutron;    pdt[p].spin2 = 1;
      pdt[p].mass = MNEUTRON;  pdt[p].mass_excess = ENEUTRON;
      break;
    case proton:
      pdt[p].za.setZA(1,1);    pdt[p].pid = proton;     pdt[p].spin2 = 1;
      pdt[p].mass = MPROTON;   pdt[p].mass_excess = EPROTON;
      break;
    case alpha:
      pdt[p].za.setZA(2,4);    pdt[p].pid = alpha;      pdt[p].spin2 = 0;
      pdt[p].mass = MALPHA;    pdt[p].mass_excess = EALPHA;
      break;
    case deuteron:
      pdt[p].za.setZA(1,2);    pdt[p].pid = deuteron;   pdt[p].spin2 = 2;
      pdt[p].mass = MDEUTERON; pdt[p].mass_excess = EDEUTERON;
      break;
    case triton:
      pdt[p].za.setZA(1,3);    pdt[p].pid = triton;     pdt[p].spin2 = 1;
      pdt[p].mass = MTRITON;   pdt[p].mass_excess = ETRITON;
      break;
    case helion:
      pdt[p].za.setZA(2,3);    pdt[p].pid = helion;     pdt[p].spin2 = 1;
      pdt[p].mass = MHELIUM3;  pdt[p].mass_excess = EHELIUM3;
      break;
    case gammaray:
      pdt[p].za.setZA(0,0);    pdt[p].pid = gammaray;   pdt[p].spin2 = 0;
      pdt[p].mass = 0.0;       pdt[p].mass_excess = 0.0;
      break;
    case userdef:
      pdt[p].za.setZA(userdefZ,userdefA);
      pdt[p].pid         = userdef;
      pdt[p].spin2       = userdefJ2;
      pdt[p].mass        = userdefMass;
      pdt[p].mass_excess = userdefMassExcess;
      break;
    default:
      pdt[p].za.setZA(0,0);    pdt[p].pid = unknown;    pdt[p].spin2 = 0;
      pdt[p].mass = 0.0;       pdt[p].mass_excess = 0.0;
      break;
    }
  }

  /*** initialize random number generator */
  if(firstcall){
    setupInitMT();
    firstcall = false;
  }
}


/**********************************************************/
/*     Setup Scattering Angles                            */
/*     -----                                              */
/*            as default, differential cross sections     */
/*            are calculated at every 1-deg in the CMS    */
/**********************************************************/
void setupScatteringAngle()
{
  double angstep = 1.0;
  for(int i=0 ; i<MAX_ANGDIST ; i++){
    crx.theta[i] = (double)(i+1)*angstep;
    crx.costh[i] = cos(crx.theta[i]*PI/180.0);
  }
}


/**********************************************************/
/*     Setup Reaction Chains, Mass, Binding Energy        */
/**********************************************************/
int setupChain(const int pmax, const double ex, ZAnumber *cZA, Pdata *pdt)
{
  int      n = 0, *gen;
  bool     found;
  double   mass, sepa, emax, eps = 1e-7;

  std::cout << "=== setupChain started ===" << std::endl;
  std::cout << "pmax: " << pmax << ", initial excitation energy: " << ex << std::endl;

  gen = new int [MAX_COMPOUND]; // generation of i-th compound

  /*** first compound */
  ncl[n].za          = *cZA;
  ncl[n].max_energy  = ex;
  ncl[n].mass_excess = mass_excess(ncl[n].za.getZ(), ncl[n].za.getA());
  gen[n] = 0;
  
  std::cout << "Added first compound: Z=" << ncl[n].za.getZ()
            << ", A=" << ncl[n].za.getA()
            << ", mass_excess=" << ncl[n].mass_excess
            << ", max_energy=" << ncl[n].max_energy << std::endl;
  n++;

  /*** add all compounds until energy becomes negative */
  for (int i = 0; i < MAX_COMPOUND; i++) {
    std::cout << "\n--- Processing compound index " << i << " ---" << std::endl;
    if (n >= MAX_COMPOUND) {
      std::cout << "Maximum number of compound nuclei reached: " << n << std::endl;
      message << "maximum number of compound nuclei reached " << n;
      cohTerminateCode("setupChain");
    }

    if ((n == i) && (i != 0)) {
      std::cout << "No new compounds added in the last iteration. Breaking out of loop." << std::endl;
      break;
    }

    for (int j = 0; j < MAX_CHANNEL; j++) {
      if (pdt[j].omp == 0) {
        std::cout << "Channel " << j << " is closed (pdt[j].omp == 0), skipping." << std::endl;
        continue;
      }
      std::cout << "Processing channel " << j << "..." << std::endl;

      ZAnumber za = ncl[i].za - pdt[j].za;
      std::cout << "Residual nucleus after channel " << j << ": Z=" << za.getZ() << ", A=" << za.getA() << std::endl;

      /*** check if residual nucleus is bigger than He */
      if ((za.getZ() <= 2) || (za.getA() <= 4)) {
        std::cout << "Residual nucleus is too light (<=He), skipping channel " << j << std::endl;
        continue;
      }

      /*** check if residual nucleus is already included */
      found = false;
      for (int k = 0; k < n; k++) {
        if (za == ncl[k].za) {
          sepa = ncl[k].mass_excess + pdt[j].mass_excess - ncl[i].mass_excess;
          emax = ncl[i].max_energy - sepa;
          std::cout << "Found duplicate candidate at index " << k 
                    << " with sepa=" << sepa 
                    << " and emax=" << emax << std::endl;
          if (fabs(1.0 - ncl[k].max_energy / emax) < eps) {
            std::cout << "Compound already included with similar energy. Marking as found." << std::endl;
            found = true;
            break;
          }
        }
      }

      /*** if this is a new nucleus, add to the NCL array */
      if (!found) {
        bool checkmass = true;
        mass = mass_excess(za.getZ(), za.getA(), &checkmass);
        sepa = mass + pdt[j].mass_excess - ncl[i].mass_excess;
        emax = ncl[i].max_energy - sepa;
        std::cout << "New compound candidate: Z=" << za.getZ() << ", A=" << za.getA()
                  << ", mass=" << mass << ", sepa=" << sepa
                  << ", emax=" << emax << std::endl;
        if (checkmass && (emax > 0.0) && (gen[i] < pmax)) {
          std::cout << "Adding new compound nucleus. Generation: " << (gen[i] + 1) << std::endl;
          cohAllocateNucleus();
          ncl[n].za.setZA(za.getZ(), za.getA());
          ncl[n].max_energy  = emax;
          ncl[n].mass_excess = mass;
          gen[n] = gen[i] + 1;
          std::cout << "Compound added at index " << n 
                    << ": Z=" << ncl[n].za.getZ() 
                    << ", A=" << ncl[n].za.getA()
                    << ", mass_excess=" << ncl[n].mass_excess 
                    << ", max_energy=" << ncl[n].max_energy 
                    << ", generation=" << gen[n] << std::endl;
          n++;
          if (n >= MAX_COMPOUND) {
            std::cout << "Reached maximum compounds while processing channel " << j << std::endl;
            break;
          }
        } else {
          std::cout << "Compound not added: checkmass=" << checkmass
                    << ", emax > 0: " << (emax > 0.0)
                    << ", gen[i] (" << gen[i] << ") < pmax (" << pmax << ")" << std::endl;
        }
      } else {
        std::cout << "Compound already exists, skipping addition for channel " << j << std::endl;
      }
    }
  }

  /*** search for all open channels */
  std::cout << "\n=== Setting up decay channels for each compound ===" << std::endl;
  for (int i = 0; i < n; i++) {
    std::cout << "\nSetting decay channels for compound index " << i << std::endl;
    for (int j = 0; j < MAX_CHANNEL; j++) {
      ncl[i].cdt[j].status         = false;
      ncl[i].cdt[j].pid            = (Particle)j;
      ncl[i].cdt[j].next           = -1;
      ncl[i].cdt[j].binding_energy = 0.0;
      ncl[i].cdt[j].spin2          = pdt[j].spin2;
      std::cout << "Initialized channel " << j << " for compound " << i << std::endl;

      ZAnumber za = ncl[i].za - pdt[j].za;
      std::cout << "Residual nucleus for channel " << j << ": Z=" << za.getZ() << ", A=" << za.getA() << std::endl;

      for (int k = 0; k < n; k++) {
        if (za == ncl[k].za) {
          sepa = ncl[k].mass_excess + pdt[j].mass_excess - ncl[i].mass_excess;
          emax = ncl[i].max_energy - sepa;
          std::cout << "Channel " << j << ": potential daughter found at index " << k 
                    << " with sepa=" << sepa 
                    << ", emax=" << emax << std::endl;
          if (fabs(1 - ncl[k].max_energy / emax) < eps) {
            ncl[i].cdt[j].next           = k;
            ncl[i].cdt[j].binding_energy = sepa;
            if (emax > 0.0)
              ncl[i].cdt[j].status = true;
            std::cout << "Channel " << j << " linked to compound " << k 
                      << " with binding energy " << sepa 
                      << " and status " << (ncl[i].cdt[j].status ? "open" : "closed") << std::endl;
            break;
          }
        }
      }
    }
    ncl[i].mass = ncl[i].za.getA() + ncl[i].mass_excess / AMUNIT;
    std::cout << "Final mass for compound " << i << ": " << ncl[i].mass << std::endl;
  }

  /*** count number of particles emitted */
  std::cout << "\n=== Counting emitted particles ===" << std::endl;
  for (int i = 0; i < MAX_COMPOUND; i++) {
    gen[i] = 0;
    for (int j = 0; j < MAX_CHANNEL; j++)
      crx.prod[i].par[j] = 0;
    crx.prod[i].xsec = crx.prod[i].fiss = 0.0;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 1; j < MAX_CHANNEL; j++) {
      int k = ncl[i].cdt[j].next;
      if (k > 0) {
        crx.prod[k] = crx.prod[i];
        crx.prod[k].par[j]++;
        std::cout << "Incrementing particle count for channel " << j 
                  << " at compound " << k << " (child of " << i << ")" << std::endl;
      }
    }
  }

  std::cout << "\n=== Finished setting up decay chain ===" << std::endl;
  std::cout << "Total number of compounds: " << n << std::endl;

  delete [] gen;

  return n;
}

/*************************************************/
/*  Count Number of Actual Compound Nuclei       */
/*************************************************/
int setupCountCompound(const int n)
{
  /*** count number of unique nuclide appears in the chain */
  int c = 0;
  for(int i=0 ; i<n ; i++){
    bool found = false;
    for(int j=0 ; j<i ; j++){
      if(ncl[i].za == ncl[j].za){
        found = true;
        break;
      }
    }
    if(!found) c++;
  }

  if(c >= MAX_NUCLIDE){
    message << "maximum number of unique nuclide reached " << c;
    cohTerminateCode("setupCountCompound");
  }

  return(c);
}


/*************************************************/
/*  Initialize MT Random Number Generator        */
/*************************************************/
void setupInitMT(void)
{
  unsigned long seed = 87545L;

#ifdef RANDOM_SEED_BY_SYSTEM
  random_device seedgen;
  seed = (unsigned long)seedgen();
#endif

  genrand_init(seed);
}
