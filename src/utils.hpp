#pragma once

arma::vec generateNeighbourVec(int N, int M);

arma::vec generateAtomTypeVec(int numAtoms, double xA);

arma::sp_mat generateHtot(int N, int M, const arma::vec &atomType,
                          const arma::vec &neighbourList);
