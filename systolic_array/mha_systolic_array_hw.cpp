#include "mha.h"
#include "systolic_array.h"
#include <hls_math.h>


void multi_head_attention(const DTYPE X[MAX_SEQ_LEN][MHA_HIDDEN_SZ], 
                            const DTYPE W_Q[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ],
                            const DTYPE W_K[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ],
                            const DTYPE W_V[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ],
                            const DTYPE W_O[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ],
                            const DTYPE W_Q_bias[MHA_HIDDEN_SZ],
                            const DTYPE W_K_bias[MHA_HIDDEN_SZ],
                            const DTYPE W_V_bias[MHA_HIDDEN_SZ],
                            const DTYPE W_O_bias[MHA_HIDDEN_SZ],
                            DTYPE O[MAX_SEQ_LEN][MHA_HIDDEN_SZ]) {
    #pragma HLS INTERFACE m_axi port=X offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=W_Q offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=W_K offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=W_V offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=W_O offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=W_Q_bias offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=W_K_bias offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=W_V_bias offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=W_O_bias offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=O offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=return


    #pragma HLS ARRAY_PARTITION variable=X dim=2 factor=4 cyclic
    #pragma HLS ARRAY_PARTITION variable=W_Q dim=1 factor=4 cyclic

    DTYPE Q[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE K[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE V[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE Concat[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    
    ComputeQ:
    matmul_l(X, W_Q, W_Q_bias, Q);

    ComputeK:
    matmul_l(X, W_K, W_K_bias, K);

    ComputeV:
    matmul_l(X, W_V, W_V_bias, V);

    Attention:
    for (int h = 0; h < NUM_HEADS; h++) {
        #pragma HLS UNROLL
        // Load matrix
        DTYPE tempQ[MAX_SEQ_LEN][HEAD_DIM];
        DTYPE tempKT[HEAD_DIM][MAX_SEQ_LEN]; // K^T
        DTYPE tempV[MAX_SEQ_LEN][HEAD_DIM];
        DTYPE tempQKT[MAX_SEQ_LEN][MAX_SEQ_LEN];
        DTYPE tempQKTV[MAX_SEQ_LEN][HEAD_DIM];
        int start_col = h * HEAD_DIM;
        for (int i = 0; i < MAX_SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                tempQ[i][j] = Q[i][start_col + j];
                tempKT[j][i] = Q[i][start_col + j];
                tempV[i][j] = Q[i][start_col + j];
            }
        }
        // QK^T
        Q_Multiply_KT:
        matmul_s(tempQ, tempKT, tempQKT);
        // Scaling
        for (int i = 0; i < MAX_SEQ_LEN; i++) {
            for (int j = 0; j < MAX_SEQ_LEN; j++) {
                tempQKT[i][j] /= 8; // Scaled Dot-Product Attention, x >> 3 = x / sqrt(HEAD_DIM)
            }
        }
        Softmax:
        for (int i = 0; i < MAX_SEQ_LEN; i++) {
            // find max value in row to avoid overflow
            DTYPE max_val = -1e9;
            for (int j = 0; j < MAX_SEQ_LEN; j++) {
                if (tempQKT[i][j] > max_val)
                    max_val = tempQKT[i][j];
            }
            DTYPE sum_exp = 0;
            for (int j = 0; j < MAX_SEQ_LEN; j++) {
                #pragma HLS PIPELINE II=3
                tempQKT[i][j] = hls::exp(tempQKT[i][j] - max_val);
                sum_exp += tempQKT[i][j];
            }
            for (int j = 0; j < MAX_SEQ_LEN; j++) {
                #pragma HLS PIPELINE II=3
                tempQKT[i][j] /= sum_exp;
            }
        }

        // Systolic Matrix Multiplication: Softmax(QK^T) * V
        // Results are concatenated.
        QKT_Multiply_V:
        matmul_s(tempQKT, tempV, tempQKTV);

        Concat:
        for (int i = 0; i < MAX_SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                Concat[i][start_col + j] = tempQKTV[i][j];
            }
        }
    }

    LinearProjection:
    matmul_l(Concat, W_O, W_O_bias, O);

}
