#pragma once

// Holds value 0 for A and 1 for B
// Generate vector to keep track of neighbours of all atoms in the lattice. The
// elements neigbourVec(4*i) - neighbourVec(4*i+3) hold the four neighbours of
// atom number i.
arma::uvec generateNeighbourVec(int N, int M);

// Generate a randomly shuffled list of all atoms in the system corresponding to
// a given x_A. Value 0 for A, 1 for B.
arma::uvec generateAtomTypeVec(int numAtoms, double xA);

// Swap two atoms in the atomType vector to create a small permutation. Return
// the swapped vector.
arma::uvec permuteAtomTypeVec(const arma::uvec &atomType, const arma::uvec &neighbourVec);

// Set up the Hamiltonian matrix as explained in the report.
// Diagonal elements will be the idividual atoms' Hamiltonian, and each column
// will also hold four more elements representing the potentials between
// neighbouring atoms.
arma::sp_mat generateHtot(int N, int M, double xA, const arma::uvec &atomType,
                          const arma::uvec &neighbourList, double r);

// Solve the Hamiltonian system and find its eigenvalues. If the Hamiltonian
// is not symmetric, something is wrong with the system and a runtime error
// is thrown.
void solveSystem(arma::vec &eigvals, const arma::sp_mat &Htot);

// Calculate the free energy $F$ of a system
double getFreeEnergy(double beta, const arma::vec &eigvals);

// Calculate the Hamming distance between two atom configurations. Considered
// to use this for the convergence criterion, but it did not work out as I hoped.
int hammingDistance(const arma::uvec &atomState1, const arma::uvec &atomState2);
