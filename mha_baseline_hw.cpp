#include "mha.h"
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
                            DTYPE O[MAX_SEQ_LEN][MHA_HIDDEN_SZ],
                            int seq_len) {
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
    #pragma HLS INTERFACE s_axilite port=seq_len bundle=control


    #pragma HLS ARRAY_PARTITION variable=X dim=2 factor=4 cyclic
    #pragma HLS ARRAY_PARTITION variable=W_Q dim=1 factor=4 cyclic

    DTYPE Q[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE K[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE V[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE QK[MAX_SEQ_LEN][MAX_SEQ_LEN];
    DTYPE Concat[MAX_SEQ_LEN][MHA_HIDDEN_SZ];

    ComputeQ:
    // Q = X*W_Q+bias
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MHA_HIDDEN_SZ; j++) {
            DTYPE sum = W_Q_bias[j];
            for (int k = 0; k < MHA_HIDDEN_SZ; k++) {
                #pragma HLS PIPELINE II=3
                sum += X[i][k] * W_Q[k][j];
            }
            Q[i][j] = sum;
        }
    }

    ComputeK:
    // K = X*W_K+bias
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MHA_HIDDEN_SZ; j++) {
            DTYPE sum = W_K_bias[j];
            for (int k = 0; k < MHA_HIDDEN_SZ; k++) {
                #pragma HLS PIPELINE II=3
                sum += X[i][k] * W_K[k][j];
            }
            K[i][j] = sum;            
        }
    }

    ComputeV:
    // V = X*W_K+bias
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MHA_HIDDEN_SZ; j++) {
            DTYPE sum = W_V_bias[j];
            for (int k = 0; k < MHA_HIDDEN_SZ; k++) {
                #pragma HLS PIPELINE II=3
                sum += X[i][k] * W_V[k][j];
            }
            V[i][j] = sum;
        }
    }

    Attention:
    for (int h = 0; h < NUM_HEADS; h++) {
        // Systolic Matrix Multiplication: Q * K^T for each head
        Q_Multiply_K:
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                for (int k = 0; k < HEAD_DIM; k++) {
                    QK[i][j] += Q[i][h * HEAD_DIM + k] * K[j][h * HEAD_DIM + k];
                }
                QK[i][j] /= 8;  // Scaled Dot-Product Attention, x >> 3 = x / sqrt(HEAD_DIM)
            }
        }
        Softmax:
        // Softmax Approximation (Row-wise)
        for (int i = 0; i < seq_len; i++) {
            // find max value in row to avoid overflow
            DTYPE max_val = -1e9;

            for (int j = 0; j < seq_len; j++) {
                if (QK[i][j] > max_val)
                    max_val = QK[i][j];
            }

            DTYPE sum_exp = 0;
            for (int j = 0; j < seq_len; j++) {
                // QK[i][j] = hls::expf(QK[i][j] - max_val);
                #pragma HLS PIPELINE II=3
                QK[i][j] = exp(QK[i][j] - max_val);
                sum_exp += QK[i][j];
            }
            for (int j = 0; j < seq_len; j++) {
                QK[i][j] /= sum_exp;
            }
        }

        // Systolic Matrix Multiplication: Softmax(QK^T) * V
        // Results are concatenated.
        QK_Multiply_V:
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                DTYPE sum = 0;
                for (int k = 0; k < seq_len; k++) {
                    #pragma HLS PIPELINE II=3
                    sum += QK[i][k] * V[k][h * HEAD_DIM + j];
                }
                Concat[i][h * HEAD_DIM + j] = sum;
            }
        }
    }

    LinearProjection:
    // Final Projection: O = Concat*W_O+bias
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MHA_HIDDEN_SZ; j++) {
            DTYPE sum = W_O_bias[j];
            for (int k = 0; k < MHA_HIDDEN_SZ; k++) {
                #pragma HLS PIPELINE II=3
                sum += Concat[i][k] * W_O[k][j];
            }
            O[i][j] = sum;
        }
    }

}