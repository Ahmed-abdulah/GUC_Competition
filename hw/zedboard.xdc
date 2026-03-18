# ============================================================
#  zedboard.xdc  —  ZedBoard (XC7Z020-CLG484-1) Constraints
#  LeNet-5 CNN Implementation
#  Vivado 2019.1
# ============================================================

# ── 100 MHz clock ────────────────────────────────────────────
set_property PACKAGE_PIN Y9       [get_ports clk]
set_property IOSTANDARD  LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name sys_clk [get_ports clk]

# ── Reset (CPU_RESET, active low) ────────────────────────────
set_property PACKAGE_PIN C9      [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# ── USB-UART (CP2104 on ZedBoard) ────────────────────────────
set_property PACKAGE_PIN W11     [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]

set_property PACKAGE_PIN T11     [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]

# ── LEDs (LD0..LD9) ──────────────────────────────────────────
set_property PACKAGE_PIN T22     [get_ports {led[0]}]
set_property PACKAGE_PIN T21     [get_ports {led[1]}]
set_property PACKAGE_PIN U22     [get_ports {led[2]}]
set_property PACKAGE_PIN U21     [get_ports {led[3]}]
set_property PACKAGE_PIN V22     [get_ports {led[4]}]
set_property PACKAGE_PIN W22     [get_ports {led[5]}]
set_property PACKAGE_PIN U19     [get_ports {led[6]}]
set_property PACKAGE_PIN U14     [get_ports {led[7]}]
set_property PACKAGE_PIN H15     [get_ports {led[8]}]
set_property PACKAGE_PIN W13     [get_ports {led[9]}]
set_property IOSTANDARD  LVCMOS33 [get_ports {led[*]}]

# ── Timing ───────────────────────────────────────────────────
set_false_path -from [get_ports rst_n]
set_false_path -from [get_ports uart_rx]
set_false_path -to   [get_ports uart_tx]
set_false_path -to   [get_ports {led[*]}]

# ── Bitstream ────────────────────────────────────────────────
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property CONFIG_VOLTAGE 3.3              [current_design]
set_property CFGBVS VCCO                     [current_design]
