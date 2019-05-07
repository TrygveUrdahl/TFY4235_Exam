#pragma once

arma::uvec monteCarloBestShuffle(int N, int M, double xA, double beta, int &iterations, int maxIterations);

arma::uvec monteCarloBestShuffleParallel(int N, int M, double xA, double beta, int maxIterations);
