#include "systolic_array.h"

DTYPE PE(hls::stream<DTYPE> &inputA, hls::stream<DTYPE> &inputB, hls::stream<DTYPE> &outputA, hls::stream<DTYPE> &outputB){
    DTYPE accum = 0;
    PE:
    for(int i = 0; i < SA_SIZE; i++){
        DTYPE A_val = inputA.read();
        DTYPE B_val = inputB.read();
        accum += A_val * B_val;
        outputA.write(A_val);
        outputB.write(B_val);
    }
    return accum;
}

DTYPE PE_H(hls::stream<DTYPE> &inputA, hls::stream<DTYPE> &inputB, hls::stream<DTYPE> &outputA){
    DTYPE accum = 0;
    PE_H:
    for(int i = 0; i < SA_SIZE; i++){
        DTYPE A_val = inputA.read();
        DTYPE B_val = inputB.read();
        accum += A_val * B_val;
        outputA.write(A_val);
    }
    return accum;
}

DTYPE PE_V(hls::stream<DTYPE> &inputA, hls::stream<DTYPE> &inputB, hls::stream<DTYPE> &outputB){
    DTYPE accum = 0;
    PE_V:
    for(int i = 0; i<SA_SIZE; i++){
        DTYPE A_val = inputA.read();
        DTYPE B_val = inputB.read();
        accum += A_val * B_val;
        outputB.write(B_val);
    }
    return accum;
}

DTYPE PE_N(hls::stream<DTYPE> &inputA, hls::stream<DTYPE> &inputB){
    DTYPE accum = 0;
    PE_N:
    for(int i = 0; i < SA_SIZE; i++){
        DTYPE A_val = inputA.read();
        DTYPE B_val = inputB.read();
        accum += A_val * B_val;
    }
    return accum;
}

void matmul_SA_SIZE(DTYPE a[SA_SIZE][SA_SIZE], DTYPE b[SA_SIZE][SA_SIZE], DTYPE out[SA_SIZE][SA_SIZE]) {
    hls::stream<DTYPE> h_fifo[SA_SIZE][SA_SIZE];
    hls::stream<DTYPE> v_fifo[SA_SIZE][SA_SIZE];
    #pragma HLS ARRAY_PARTITION variable = a dim = 2 complete
    #pragma HLS ARRAY_PARTITION variable = b dim = 1 complete
    #pragma HLS ARRAY_PARTITION variable = out dim = 0 complete
    // initialize
    for (int i = 0; i < SA_SIZE; ++i) {
        #pragma HLS UNROLL
        for (size_t j = 0; j < SA_SIZE; ++j) {
            #pragma HLS UNROLL
            h_fifo[i][0].write(a[i][j]);
            v_fifo[0][j].write(b[i][j]);
        }
    }

    #pragma HLS DATAFLOW
    matmulSMALL:
    for(int i = 0; i < SA_SIZE; i++) {
        #pragma HLS UNROLL
        for(int j = 0; j < SA_SIZE; j++) {
            #pragma HLS UNROLL
            if(i + 1 < SA_SIZE && j + 1 < SA_SIZE) {
                out[i][j] = PE(h_fifo[i][j], v_fifo[i][j], h_fifo[i][j+1], v_fifo[i+1][j]);
            } else if (i + 1 >= SA_SIZE && j + 1 < SA_SIZE ) {
                out[i][j] = PE_H(h_fifo[i][j], v_fifo[i][j], h_fifo[i][j+1]);
            } else if (i + 1 < SA_SIZE && j + 1 >= SA_SIZE ) {
                out[i][j] = PE_V(h_fifo[i][j], v_fifo[i][j], v_fifo[i+1][j]);
            } else {
                out[i][j] = PE_N(h_fifo[i][j], v_fifo[i][j]);
            }
            
        }
    }
}

void matmul_s(
    const DTYPE A[SIZE_S][SIZE_S],
    const DTYPE B[SIZE_S][SIZE_S],
    DTYPE Out[SIZE_S][SIZE_S]
) {
    DTYPE tempA[SA_SIZE][SA_SIZE];
    DTYPE tempB[SA_SIZE][SA_SIZE];
    DTYPE tempOut[SA_SIZE][SA_SIZE];

    // Output Stationary
    for(int rowO = 0; rowO < SIZE_S; rowO += SA_SIZE) {
        for(int colO = 0; colO < SIZE_L; colO += SA_SIZE) {
            for(int k = 0; k < SIZE_L; k += SA_SIZE) {
                // Load matrix
                for (int i = 0; i < SA_SIZE; i++) {
                    for (int j = 0; j < SA_SIZE; j++) {
                        #pragma HLS PIPELINE II=3
                        tempA[i][j] = A[rowO + i][k + j];
                        tempB[i][j] = B[k + i][colO + j];
                        tempOut[i][j] = 0;
                    }
                }
                // Compute
                matmul_SA_SIZE(tempA, tempB, tempOut);
                // Load to Output
                for (int i = 0; i < SA_SIZE; i++) {
                    for (int j = 0; j < SA_SIZE; j++) {
                        #pragma HLS PIPELINE II=3
                        Out[rowO + i][colO + j] += tempOut[i][j];
                    }
                }
            }
        }
    }
}

void matmul_l(
    const DTYPE A[SIZE_S][SIZE_L],
    const DTYPE B[SIZE_L][SIZE_L],
    const DTYPE Bias[SIZE_L],
    DTYPE Out[SIZE_S][SIZE_L]
) {
    // Load Bias
    for(int i = 0; i < SIZE_S; i++) {
        for(int j = 0; j < SIZE_L; j++) {
            Out[i][j] = Bias[j];
        }
    }

    DTYPE tempA[SA_SIZE][SA_SIZE];
    DTYPE tempB[SA_SIZE][SA_SIZE];
    DTYPE tempOut[SA_SIZE][SA_SIZE];

    // Output Stationary
    for(int rowO = 0; rowO < SIZE_S; rowO += SA_SIZE) {
        for(int colO = 0; colO < SIZE_L; colO += SA_SIZE) {
            for(int k = 0; k < SIZE_L; k += SA_SIZE) {
                // Load matrix
                for (int i = 0; i < SA_SIZE; i++) {
                    for (int j = 0; j < SA_SIZE; j++) {
                        #pragma HLS PIPELINE II=3
                        tempA[i][j] = A[rowO + i][k + j];
                        tempB[i][j] = B[k + i][colO + j];
                        tempOut[i][j] = 0;
                    }
                }
                // Compute
                matmul_SA_SIZE(tempA, tempB, tempOut);
                // Load to Output
                for (int i = 0; i < SA_SIZE; i++) {
                    for (int j = 0; j < SA_SIZE; j++) {
                        #pragma HLS PIPELINE II=3
                        Out[rowO + i][colO + j] += tempOut[i][j];
                    }
                }
            }
        }
    }
}