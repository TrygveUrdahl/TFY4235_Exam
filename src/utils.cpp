#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>

// value 0 for A, 1 for B
arma::vec generateAtomTypeVec(int numAtoms, double xA) {
  int numA = numAtoms * xA;
  arma::vec atomType(numAtoms);
  int usedA = 0;

  for (int i = 0; i < numAtoms; i++) {
    if (usedA < numA) {
      usedA++;
      atomType(i) = 0;
    }
    else {
      atomType(i) = 1;
    }
  }
  return arma::shuffle(atomType);
}

arma::vec generateNeighbourVec(int N, int M) {
  int numAtoms = N * M;
  int numNeighbours = 4 * numAtoms;
  arma::vec neighbourVec(numNeighbours, arma::fill::zeros);

  for (int i = 0; i < numAtoms; i++) {
    neighbourVec(4 * i + 0) = i - M; // Top
    neighbourVec(4 * i + 1) = i + 1; // Right
    neighbourVec(4 * i + 2) = i + M; // Bottom
    neighbourVec(4 * i + 3) = i - 1; // Left

    if ((i - M) < 0) { // Top loop-around
      neighbourVec(4 * i + 0) = i + numAtoms - M;
    }
    if ((i + 1) % M == 0) { // Right loop-around
      neighbourVec(4 * i + 1) = (i - M + 1);
    }
    if ((i + M) > numAtoms - 1) { // Bottom loop-around
      neighbourVec(4 * i + 2) = i % M;
    }
    if ((((i + M) - 1) % M) == M - 1) { // Left loop-around
      neighbourVec(4 * i + 3) = (i + M - 1);
    }
  }
  return neighbourVec;
}


arma::sp_mat generateHtot(int N, int M, double xA) {
  int numAtoms = N * M;
  arma::sp_mat Htot(numAtoms, numAtoms);
  arma::vec atomType = generateAtomTypeVec(numAtoms, xA);

  for (int i = 0; i < numAtoms; i++) {
    int idx = i % M;
    int idy = i / M;
  }
  return Htot;
}
