import numpy as np
import pynq
from pynq import Overlay, Device
import time

device = Device.devices[0]
overlay = Overlay("/home/xilinx/jupyter_notebooks/mha_sa/mha_sa.bit", device = device)

ip = overlay.multi_head_attention

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

X_fixed = np.vectorize(float_to_fixed)(X_float)
W_Q_fixed = np.vectorize(float_to_fixed)(W_Q_float)
W_K_fixed = np.vectorize(float_to_fixed)(W_K_float)
W_V_fixed = np.vectorize(float_to_fixed)(W_V_float)
W_O_fixed = np.vectorize(float_to_fixed)(W_O_float)
W_Q_bias_fixed = np.vectorize(float_to_fixed)(W_Q_bias_float)
W_K_bias_fixed = np.vectorize(float_to_fixed)(W_K_bias_float)
W_V_bias_fixed = np.vectorize(float_to_fixed)(W_V_bias_float)
W_O_bias_fixed = np.vectorize(float_to_fixed)(W_O_bias_float)

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

np.copyto(X_buffer, X_fixed)
np.copyto(W_Q_buffer, W_Q_fixed)
np.copyto(W_K_buffer, W_K_fixed)
np.copyto(W_V_buffer, W_V_fixed)
np.copyto(W_O_buffer, W_O_fixed)
np.copyto(W_Q_bias_buffer, W_Q_bias_fixed)
np.copyto(W_K_bias_buffer, W_K_bias_fixed)
np.copyto(W_V_bias_buffer, W_V_bias_fixed)
np.copyto(W_O_bias_buffer, W_O_bias_fixed)

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
ip.mmio.write_reg(0x18, W_Q_buffer.physical_address)
ip.mmio.write_reg(0x1C, 0)
ip.mmio.write_reg(0x20, W_K_buffer.physical_address)
ip.mmio.write_reg(0x24, 0)
ip.mmio.write_reg(0x28, W_V_buffer.physical_address)
ip.mmio.write_reg(0x2C, 0)
ip.mmio.write_reg(0x30, W_O_buffer.physical_address)
ip.mmio.write_reg(0x34, 0)
ip.mmio.write_reg(0x38, W_Q_bias_buffer.physical_address)
ip.mmio.write_reg(0x3C, 0)
ip.mmio.write_reg(0x40, W_K_bias_buffer.physical_address)
ip.mmio.write_reg(0x44, 0)
ip.mmio.write_reg(0x48, W_V_bias_buffer.physical_address)
ip.mmio.write_reg(0x4C, 0)
ip.mmio.write_reg(0x50, W_O_bias_buffer.physical_address)
ip.mmio.write_reg(0x54, 0)
ip.mmio.write_reg(0x58, O_buffer.physical_address)
ip.mmio.write_reg(0x5C, 0)

ip_status = ip.read(0x00)
print(ip_status)

# Start execution
ip.write(0x00, 1)
ip_status = ip.read(0x00)
print(ip_status)

# Wait until finish
while (ip_status == 14):
    ip_status = ip.read(0x00)

ip_status = ip.read(0x00)
print(ip_status)
print('Accelerator Done!')

O_buffer.sync_from_device()
O_float = np.vectorize(fixed_to_float)(O_buffer)
print("Output: \n", O_float)

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