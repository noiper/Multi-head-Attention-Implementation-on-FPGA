# Multi-head-Attention-Implementation-on-FPGA

This is the final project for EECS 221.


## File Usage
mha.h: common header defining model parameters

test_mha.cpp: testbench

mha_baseline_hw.cpp: baseline implementation.

mha_block_hw.cpp: block matrix multiplication implementation.

systolic_array/*: systolic array implementation.

PYNQ/: related files to run the accelerators on board with PYNQ
