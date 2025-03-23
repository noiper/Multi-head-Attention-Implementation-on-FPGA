#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "mha.h"
#include "hls_stream.h"

#define SA_SIZE 8

// Small matmul (64*64)*(64*64)=(64*64)
// Large matmul (64*128)*(128*128)=(64*128)
#define SIZE_S 64
#define SIZE_L 128

// (16 * 16) * (16 * 16)
void matmul_SA_SIZE(
    DTYPE a[SA_SIZE][SA_SIZE],
    DTYPE b[SA_SIZE][SA_SIZE],
    DTYPE out[SA_SIZE][SA_SIZE]
);

// (64 * 64) * (64 * 64)
void matmul_s(
    const DTYPE A[SIZE_S][SIZE_S],
    const DTYPE B[SIZE_S][SIZE_S],
    DTYPE Out[SIZE_S][SIZE_S]
);

// (64 * 128) * (128 * 128)
void matmul_l(
    const DTYPE A[SIZE_S][SIZE_L],
    const DTYPE B[SIZE_L][SIZE_L],
    const DTYPE Bias[SIZE_L],
    DTYPE Out[SIZE_S][SIZE_L]
);

#endif
