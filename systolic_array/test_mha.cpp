#include "mha.h"
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <unistd.h>
#include <limits.h>  // For PATH_MAX

using namespace std;

void multi_head_attention_sw(const DTYPE X[MAX_SEQ_LEN][MHA_HIDDEN_SZ], 
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

    DTYPE Q[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE K[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE V[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE QK[MAX_SEQ_LEN][MAX_SEQ_LEN];
    DTYPE Concat[MAX_SEQ_LEN][MHA_HIDDEN_SZ];

    // Q = X*W_Q+bias
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MHA_HIDDEN_SZ; j++) {
            DTYPE sum = W_Q_bias[j];
            for (int k = 0; k < MHA_HIDDEN_SZ; k++) {
                sum += X[i][k] * W_Q[k][j];
            }
            Q[i][j] = sum;
        }
    }
    // K = X*W_K+bias
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MHA_HIDDEN_SZ; j++) {
            DTYPE sum = W_K_bias[j];
            for (int k = 0; k < MHA_HIDDEN_SZ; k++) {
                sum += X[i][k] * W_K[k][j];
            }
            K[i][j] = sum;            
        }
    }
    // V = X*W_K+bias
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MHA_HIDDEN_SZ; j++) {
            DTYPE sum = W_V_bias[j];
            for (int k = 0; k < MHA_HIDDEN_SZ; k++) {
                sum += X[i][k] * W_V[k][j];
            }
            V[i][j] = sum;
        }
    }

    for (int h = 0; h < NUM_HEADS; h++) {
        // Systolic Matrix Multiplication: Q * K^T for each head
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                for (int k = 0; k < HEAD_DIM; k++) {
                    QK[i][j] += Q[i][h * HEAD_DIM + k] * K[j][h * HEAD_DIM + k];
                }
                QK[i][j] /= 8;  // Scaled Dot-Product Attention, x >> 3 = x / sqrt(HEAD_DIM)
            }
        }

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
                QK[i][j] = exp((float)(QK[i][j] - max_val));
                sum_exp += QK[i][j];
            }
            for (int j = 0; j < seq_len; j++) {
                QK[i][j] /= sum_exp;
            }
        }

        // Systolic Matrix Multiplication: Softmax(QK^T) * V
        // Results are concatenated.
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                DTYPE sum = 0;
                for (int k = 0; k < seq_len; k++) {
                    sum += QK[i][k] * V[k][h * HEAD_DIM + j];
                }
                Concat[i][h * HEAD_DIM + j] = sum;
            }
        }
    }

    // Final Projection: O = Concat*W_O+bias
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < MHA_HIDDEN_SZ; j++) {
            DTYPE sum = W_O_bias[j];
            for (int k = 0; k < MHA_HIDDEN_SZ; k++) {
                sum += Concat[i][k] * W_O[k][j];
            }
            O[i][j] = sum;
        }
    }

}
    
void init_rand_matrices(DTYPE *mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            mat[i * cols + j] = (DTYPE)(rand() % 512);
        }
    }
}

int main()
{
    int fail = 0;

    DTYPE X[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE W_Q[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ];
    DTYPE W_K[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ];
    DTYPE W_V[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ];
    DTYPE W_O[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ];
    DTYPE W_Q_bias[MHA_HIDDEN_SZ];
    DTYPE W_K_bias[MHA_HIDDEN_SZ];
    DTYPE W_V_bias[MHA_HIDDEN_SZ];
    DTYPE W_O_bias[MHA_HIDDEN_SZ];

    DTYPE hw_O[MAX_SEQ_LEN][MHA_HIDDEN_SZ];
    DTYPE sw_O[MAX_SEQ_LEN][MHA_HIDDEN_SZ];

    initWeightMatrices:
    init_rand_matrices((DTYPE *)W_Q, MHA_HIDDEN_SZ, MHA_HIDDEN_SZ);
    init_rand_matrices((DTYPE *)W_K, MHA_HIDDEN_SZ, MHA_HIDDEN_SZ);
    init_rand_matrices((DTYPE *)W_V, MHA_HIDDEN_SZ, MHA_HIDDEN_SZ);
    init_rand_matrices((DTYPE *)W_O, MHA_HIDDEN_SZ, MHA_HIDDEN_SZ);
    init_rand_matrices((DTYPE *)W_Q_bias, MHA_HIDDEN_SZ, 1);
    init_rand_matrices((DTYPE *)W_K_bias, MHA_HIDDEN_SZ, 1);
    init_rand_matrices((DTYPE *)W_V_bias, MHA_HIDDEN_SZ, 1);
    init_rand_matrices((DTYPE *)W_O_bias, MHA_HIDDEN_SZ, 1);

    initInputOutput:
    init_rand_matrices((DTYPE *)X, MAX_SEQ_LEN, MHA_HIDDEN_SZ);
    init_rand_matrices((DTYPE *)hw_O, MAX_SEQ_LEN, MHA_HIDDEN_SZ);
    init_rand_matrices((DTYPE *)sw_O, MAX_SEQ_LEN, MHA_HIDDEN_SZ);
    
    int seq_len = rand() % MAX_SEQ_LEN;
    if (seq_len == 0)
        seq_len = 1;
    multi_head_attention(X, W_Q, W_K, W_V, W_O, W_Q_bias, W_K_bias, W_V_bias, W_O_bias, hw_O);
    multi_head_attention_sw(X, W_Q, W_K, W_V, W_O, W_Q_bias, W_K_bias, W_V_bias, W_O_bias, sw_O, seq_len);

    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < MHA_HIDDEN_SZ; j++)
        {
            if (hw_O[i][j] != sw_O[i][j])
            {
                fail = 1;
                printf("Mismatch at %d (%d), %d: HW = %f, SW = %f\n", i, seq_len, j, (float)hw_O[i][j], (float)sw_O[i][j]);

            }
        }
    }

    if (fail == 1)
        printf("FAILED\n");
    else
        printf("PASS\n");
    return fail;
}
