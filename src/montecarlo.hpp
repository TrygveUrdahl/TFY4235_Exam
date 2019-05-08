#pragma once

arma::uvec monteCarloBestShuffle(int N, int M, double xA, double beta, int &iterations, int maxIterations, double r);

arma::uvec monteCarloBestShuffleParallel(int N, int M, double xA, double beta, int maxIterations, double r);
