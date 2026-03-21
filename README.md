# LeNet-5 CNN on FPGA — ZedBoard XC7Z020

Hardware implementation of LeNet-5 for handwritten digit recognition (MNIST), built for the **IEEE CASS Seasonal School 2025 Digital Hardware Competition**.

Covers the full stack: PyTorch training → Q6 fixed-point quantization → Verilog RTL → QuestaSim simulation → Vivado synthesis on Xilinx XC7Z020.

**Result: 10/10 digits correctly classified in RTL, matching the Python integer reference exactly.**

---

## What's in here

```
hw/
├── lenet5_top_v2.v              top-level (UART, BRAM mux, master FSM, argmax)
├── conv_maxpool_block_final.v   CONV1 + CONV2 + MaxPool (serial MAC)
├── fc_control_v2.v              FC1, FC2, FC3 controller
├── mac_unit_final.v             pipelined MAC unit
├── fsm_control_final.v          master pipeline FSM
├── bram_tdp.v                   8192×8 true dual-port BRAM
├── rom_module.v                 parameterised weight/bias ROM
├── uart.v                       UART RX/TX at 115200 baud
├── pe_block.v                   PE block (ready for parallel upgrade)
├── pe_matrix.v                  PE matrix (ready for parallel upgrade)
├── pe_matrix_ctrl.v             PE controller (ready for parallel upgrade)
├── zedboard.xdc                 timing + pin constraints
├── hex_weights/                 quantized weights (10 × .hex)
├── tb_10digits.v                10-digit verification testbench
└── ref_hw_inference.py          integer golden reference model

Lenet5.py                        PyTorch training
weight_export.py                 weight quantization → hex files
image_to_sim.py                  image preprocessing for simulation
prepare_10digits.py              batch prep for all 10 digits
trace_layers.py                  layer-by-layer output tracing
```

---

## How to run

**1. Train and export weights**
```bash
python Lenet5.py
python weight_export.py
```

**2. Prepare test images** (put `0.png`..`9.png` in `hw/`)
```bash
cd hw
python prepare_10digits.py
```
This runs the Python integer reference on each image and writes `golden.txt`.

**3. Simulate**
```bash
cd hw
vdel -all -lib work && vlib work
vlog bram_tdp.v rom_module.v ram_module.v pe_block.v pe_matrix.v pe_matrix_ctrl.v \
     mac_unit_final.v conv_maxpool_block_final.v fc_control_v2.v \
     fsm_control_final.v uart.v lenet5_top_v2.v tb_10digits.v
vsim -c -do "run -all; quit" work.tb_10 2>&1 | tee sim_results.log
```

**4. Verify**
```bash
python compare_rtl_vs_ref.py
```

---

## Architecture

```
UART RX (784 bytes)
    │
    ▼
Feature Map BRAM  (8192 × 8-bit, single shared memory for all layers)
    │
    ├── CONV1  6 filters 5×5  28×28 → 24×24
    ├── POOL1  2×2 max          24×24 → 12×12
    ├── CONV2  16 filters 5×5  12×12 → 8×8   (6 serial batches)
    ├── POOL2  2×2 max           8×8  → 4×4
    ├── FC1    256 → 120  ReLU
    ├── FC2    120 →  84  ReLU
    └── FC3     84 →  10  (raw logits)
    │
    ▼
Argmax → UART TX + LED
```

All arithmetic is **8-bit signed Q6 fixed-point** (6 fractional bits, scale = 64).

### Note on the PE matrix files

The paper specifies a 2D systolic PE array for convolution. Those files (`pe_block.v`, `pe_matrix.v`, `pe_matrix_ctrl.v`) are written and in the repo. The current implementation uses a serial single-MAC loop instead — this made verification practical and allowed finding all 7 bugs during development. The PE files are ready to wire in when throughput becomes a priority.

---

## Synthesis (Vivado 2019.1 — XC7Z020)

| Resource | Used | Utilization |
|----------|------|-------------|
| LUTs | 9,189 | 3.03% |
| Flip-Flops | 6,968 | 1.15% |
| Block RAMs | 2.50 | 1.79% |
| DSPs | 2 | 0.91% |

---

## Authors

Ahmed Abdullah Abdelmonem Ibrahim  
Mohammed Wael Mounir  
Mohammed Ali Thabet

German University in Cairo — IEEE CASS Seasonal School 2025
