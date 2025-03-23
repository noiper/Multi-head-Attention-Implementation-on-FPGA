import numpy as np
import pynq
from pynq import Overlay, Device
import time

device = Device.devices[0]
overlay = Overlay("/home/xilinx/jupyter_notebooks/mha_sa/mha_sa.bit", device = device)

ip = overlay.multi_head_attention_0
# ip = overlay.pl_vecadd_0

print(ip.register_map)


def float_to_fixed(value, total_bits=8, int_bits=4):
    frac_bits = total_bits - int_bits
    return np.int8(value * (1 << frac_bits))

def fixed_to_float(value, total_bits=8, int_bits=4):
    frac_bits = total_bits - int_bits
    return value / (1 << frac_bits)

S, L = 64, 128
X_float = np.random.uniform(-8, 7.9375, (S, L)).astype(np.float32)
W_Q_float = np.random.uniform(-8, 7.9375, (L, L)).astype(np.float32)
W_K_float = np.random.uniform(-8, 7.9375, (L, L)).astype(np.float32)
W_V_float = np.random.uniform(-8, 7.9375, (L, L)).astype(np.float32)
W_O_float = np.random.uniform(-8, 7.9375, (L, L)).astype(np.float32)
W_Q_bias_float = np.random.uniform(-8, 7.9375, (L,)).astype(np.float32)
W_K_bias_float = np.random.uniform(-8, 7.9375, (L,)).astype(np.float32)
W_V_bias_float = np.random.uniform(-8, 7.9375, (L,)).astype(np.float32)
W_O_bias_float = np.random.uniform(-8, 7.9375, (L,)).astype(np.float32)
# Convert to fixed point
X_fixed = np.vectorize(float_to_fixed)(X_float)
W_Q_fixed = np.vectorize(float_to_fixed)(W_Q_float)
W_K_fixed = np.vectorize(float_to_fixed)(W_K_float)
W_V_fixed = np.vectorize(float_to_fixed)(W_V_float)
W_O_fixed = np.vectorize(float_to_fixed)(W_O_float)
W_Q_bias_fixed = np.vectorize(float_to_fixed)(W_Q_bias_float)
W_K_bias_fixed = np.vectorize(float_to_fixed)(W_K_bias_float)
W_V_bias_fixed = np.vectorize(float_to_fixed)(W_V_bias_float)
W_O_bias_fixed = np.vectorize(float_to_fixed)(W_O_bias_float)
# Create buffer that FPGA can access
X_buffer = pynq.allocate(shape=(S, L), dtype=np.int8)
W_Q_buffer = pynq.allocate(shape=(L, L), dtype=np.int8)
W_K_buffer = pynq.allocate(shape=(L, L), dtype=np.int8)
W_V_buffer = pynq.allocate(shape=(L, L), dtype=np.int8)
W_O_buffer = pynq.allocate(shape=(L, L), dtype=np.int8)
W_Q_bias_buffer = pynq.allocate(shape=(L,), dtype=np.int8)
W_K_bias_buffer = pynq.allocate(shape=(L,), dtype=np.int8)
W_V_bias_buffer = pynq.allocate(shape=(L,), dtype=np.int8)
W_O_bias_buffer = pynq.allocate(shape=(L,), dtype=np.int8)
O_buffer = pynq.allocate(shape=(S, L), dtype=np.int8)
# Load data into buffer
np.copyto(X_buffer, X_fixed)
np.copyto(W_Q_buffer, W_Q_fixed)
np.copyto(W_K_buffer, W_K_fixed)
np.copyto(W_V_buffer, W_V_fixed)
np.copyto(W_O_buffer, W_O_fixed)
np.copyto(W_Q_bias_buffer, W_Q_bias_fixed)
np.copyto(W_K_bias_buffer, W_K_bias_fixed)
np.copyto(W_V_bias_buffer, W_V_bias_fixed)
np.copyto(W_O_bias_buffer, W_O_bias_fixed)
# Sync to device
X_buffer.sync_to_device()
W_Q_buffer.sync_to_device()
W_K_buffer.sync_to_device()
W_V_buffer.sync_to_device()
W_O_buffer.sync_to_device()
W_Q_bias_buffer.sync_to_device()
W_K_bias_buffer.sync_to_device()
W_V_bias_buffer.sync_to_device()
W_O_bias_buffer.sync_to_device()
O_buffer.sync_to_device()

ip.mmio.write_reg(0x10, X_buffer.physical_address)
ip.mmio.write_reg(0x14, 0)
ip.mmio.write_reg(0x18, 0)
ip.mmio.write_reg(0x1C, 0)
ip.mmio.write_reg(0x20, 0)
ip.mmio.write_reg(0x24, 0)
ip.mmio.write_reg(0x28, 0)
ip.mmio.write_reg(0x2C, 0)
# 30 reserved
ip.mmio.write_reg(0x34, W_Q_buffer.physical_address)
ip.mmio.write_reg(0x38, 0)
ip.mmio.write_reg(0x3C, 0)
ip.mmio.write_reg(0x40, 0)
ip.mmio.write_reg(0x44, 0)
ip.mmio.write_reg(0x48, 0)
ip.mmio.write_reg(0x4C, 0)
ip.mmio.write_reg(0x50, 0)
# 54 reserved
ip.mmio.write_reg(0x58, W_K_buffer.physical_address)
ip.mmio.write_reg(0x5C, 0)
# 60 reserved
ip.mmio.write_reg(0x64, W_V_buffer.physical_address)
ip.mmio.write_reg(0x68, 0)
# 6C reserved
ip.mmio.write_reg(0x70, W_O_buffer.physical_address)
ip.mmio.write_reg(0x74, 0)
# 78 reserved
ip.mmio.write_reg(0x7C, W_Q_bias_buffer.physical_address)
ip.mmio.write_reg(0x80, 0)
# 84 reserved
ip.mmio.write_reg(0x88, W_K_bias_buffer.physical_address)
ip.mmio.write_reg(0x8C, 0)
# 90 reserved
ip.mmio.write_reg(0x94, W_V_bias_buffer.physical_address)
ip.mmio.write_reg(0x98, 0)
# 9C reserved
ip.mmio.write_reg(0xA0, W_O_bias_buffer.physical_address)
ip.mmio.write_reg(0xA4, 0)
# A8 reserved
ip.mmio.write_reg(0xAC, O_buffer.physical_address)
ip.mmio.write_reg(0xB0, 0)

ip_status = ip.read(0x00)
print(ip_status)

# Start execution
ip.write(0x00, 14)
ip_status = ip.read(0x00)
print(ip_status)

while (ip_status == 1):
    print(ip_status)
    ip_status = ip.read(0x00)

# Wait until finish
while (ip_status == 14):
    ip_status = ip.read(0x00)

ip_status = ip.read(0x00)
print(ip_status)
print('Acceleration Done!')

O_buffer.sync_from_device()
O_float = np.vectorize(fixed_to_float)(O_buffer)
print("Output: \n", O_float)

X_buffer.sync_from_device()
X_out = np.vectorize(fixed_to_float)(X_buffer)
print("X: \n", X_out)

# Free buffer
X_buffer.freebuffer()
W_Q_buffer.freebuffer()
W_K_buffer.freebuffer()
W_V_buffer.freebuffer()
W_O_buffer.freebuffer()
W_Q_bias_buffer.freebuffer()
W_K_bias_buffer.freebuffer()
W_V_bias_buffer.freebuffer()
W_O_bias_buffer.freebuffer()
O_buffer.freebuffer()
