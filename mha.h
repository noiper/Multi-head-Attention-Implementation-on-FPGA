#ifndef __MHA_H__
#define __MHA_H__

#include "ap_fixed.h"
#define MHA_LAYER 2
#define MHA_HIDDEN_SZ 128
#define NUM_HEADS 2
#define HEAD_DIM (MHA_HIDDEN_SZ / NUM_HEADS)
#define FFN_HIDDEN_SZ 512
#define MAX_SEQ_LEN 64   // Maximum sequence length

typedef ap_fixed<8, 4, AP_RND, AP_SAT> DTYPE;

void multi_head_attention(const DTYPE X[MAX_SEQ_LEN][MHA_HIDDEN_SZ], 
    const DTYPE W_Q[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ],
    const DTYPE W_K[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ],
    const DTYPE W_V[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ],
    const DTYPE W_O[MHA_HIDDEN_SZ][MHA_HIDDEN_SZ],
    const DTYPE W_Q_bias[MHA_HIDDEN_SZ],
    const DTYPE W_K_bias[MHA_HIDDEN_SZ],
    const DTYPE W_V_bias[MHA_HIDDEN_SZ],
    const DTYPE W_O_bias[MHA_HIDDEN_SZ],
    DTYPE O[MAX_SEQ_LEN][MHA_HIDDEN_SZ]);

#endif // __MHA_H__
