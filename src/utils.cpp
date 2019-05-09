#include <iostream>
#include <cmath>
#include <armadillo>
#include <chrono>
#include <omp.h>

#include "montecarlo.hpp"

// Generate a randomly shuffled list of all atoms in the system corresponding to
// a given x_A. Value 0 for A, 1 for B.
arma::uvec generateAtomTypeVec(int numAtoms, double xA) {
  const int numA = std::round(numAtoms * xA);
  arma::uvec atomType(numAtoms);
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

// Swap two atoms in the atomType vector to create a small permutation. Return
// the swapped vector.
arma::uvec permuteAtomTypeVec(const arma::uvec &atomType, const arma::uvec &neighbourVec) {
  arma::uvec atomTypePermuted = atomType;
  int index = arma::randi<int>(arma::distr_param(0, atomType.n_elem - 1));
  int swap_index = arma::randi<int>(arma::distr_param(0, 3));
  std::swap(atomTypePermuted(index), atomTypePermuted(neighbourVec(4*index + swap_index)));
  return atomTypePermuted;
}

// Holds value 0 for A and 1 for B
// Generate vector to keep track of neighbours of all atoms in the lattice. The
// elements neigbourVec(4*i) - neighbourVec(4*i+3) hold the four neighbours of
// atom number i.
arma::uvec generateNeighbourVec(int N, int M) {
  const int numAtoms = N * M;
  const int numNeighbours = 4 * numAtoms;
  arma::uvec neighbourVec(numNeighbours, arma::fill::zeros);

  for (int i = 0; i < numAtoms; i++) {
    neighbourVec(4 * i + 0) = i - M; // Top
    neighbourVec(4 * i + 1) = i + 1; // Right
    neighbourVec(4 * i + 2) = i + M; // Bottom
    neighbourVec(4 * i + 3) = i - 1; // Left

    // Check all edge cases
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

// Return the potential between two atoms depending on what types they are
double V(double r, int atom1, int atom2) {
  if (atom1 == 0 && atom2 == 0) {
    constexpr double epsilonAA = 5.0;
    constexpr double gammaAA = 0.80;
    constexpr double etaAA = 1.95;
    const double value = epsilonAA * (std::pow(gammaAA/r, 6)  - std::exp(-r/etaAA));
    return value; // $V_{AA}$
  }
  else if (atom1 == 1 && atom2 == 1) {
    constexpr double epsilonBB = 6.0;
    constexpr double gammaBB = 0.70;
    constexpr double etaBB = 0.70;
    const double value = epsilonBB * (std::pow(gammaBB/r, 6)  - std::exp(-r/etaBB));
    return value; // $V_{BB}$
  }
  else {
    constexpr double epsilonAB = 4.5;
    constexpr double gammaAB = 0.85;
    constexpr double etaAB = 1.90;
    const double value = epsilonAB * (std::pow(gammaAB/r, 6)  - std::exp(-r/etaAB));
    return value; // $V_{AB}$
  }
}

// Return the Hamiltonian of atom A or B
double H(int atom1) {
  if (atom1 == 0) {
    return 0.20; // $H_A$
  }
  else {
    return 0.25; // $H_B$
  }
}

// Set up the Hamiltonian matrix as explained in the report.
// Diagonal elements will be the idividual atoms' Hamiltonian, and each column
// will also hold four more elements representing the potentials between
// neighbouring atoms.
arma::sp_mat generateHtot(int N, int M, double xA, const arma::uvec &atomType,
                          const arma::uvec &neighbourList, double r) {
  const int numAtoms = N * M;
  arma::sp_mat Htot(numAtoms, numAtoms);
  for (int i = 0; i < numAtoms; i++) {
    const int currentAtom = atomType(i);
    Htot(i,                        i) = H(currentAtom); // $H_n$
    Htot(i, neighbourList(4 * i + 0)) = V(r, currentAtom, atomType(neighbourList(4 * i + 0))); // $V_{nn}$
    Htot(i, neighbourList(4 * i + 1)) = V(r, currentAtom, atomType(neighbourList(4 * i + 1))); // $V_{nn}$
    Htot(i, neighbourList(4 * i + 2)) = V(r, currentAtom, atomType(neighbourList(4 * i + 2))); // $V_{nn}$
    Htot(i, neighbourList(4 * i + 3)) = V(r, currentAtom, atomType(neighbourList(4 * i + 3))); // $V_{nn}$
  }
  if (!Htot.is_symmetric()) {
    throw std::runtime_error("Hamiltonian is not symmetric! ");
  }
  return Htot;
}

// Solve the Hamiltonian system and find its eigenvalues. If the Hamiltonian
// is not symmetric, something is wrong with the system and a runtime error
// is thrown.
void solveSystem(arma::vec &eigvals, const arma::sp_mat &Htot) {
  if (Htot.is_symmetric()) {
    arma::eig_sym(eigvals, arma::mat(Htot));
  }
  else {
    throw std::runtime_error("Hamiltonian is not symmetric! ");
  }
}

// Calculate the free energy $F$ of a system
double getFreeEnergy(double beta, const arma::vec &eigvals) {
  double freeEnergy = 0;
  // #pragma omp parallel for schedule(static) reduction(+:freeEnergy) // Causes slow-down because of overhead
  for (int i = 0; i < eigvals.n_elem; i++) {
    freeEnergy += std::log(1.0 + std::exp(-beta * eigvals(i)));
  }
  freeEnergy *= -1.0/beta;
  if (freeEnergy > 0) {
    throw std::runtime_error("Free energy is positive, something went wrong. ");
  }
  return freeEnergy;
}


// Calculate the Hamming distance between two atom configurations. Considered
// to use this for the convergence criterion, but it did not work out as I hoped.
int hammingDistance(const arma::uvec &atomState1, const arma::uvec &atomState2) {
  int dist = 0;
  if (atomState1.n_elem != atomState2.n_elem) {
    return std::numeric_limits<int>::max();
  }
  for (int i = 0; i <atomState1.n_elem; i++) {
    if (atomState1(i) != atomState2(i)) {
      dist++;
    }
  }
  return dist;
}
