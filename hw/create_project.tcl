# ============================================================
#  create_project.tcl  —  Vivado 2019.1 Project Setup Script
#  ZedBoard XC7Z020-CLG484-1
#  Run: vivado -mode batch -source create_project.tcl
# ============================================================

# ── Create project ───────────────────────────────────────────
create_project lenet5_zedboard ./vivado_project -part xc7z020clg484-1
set_property board_part em.avnet.com:zed:part0:1.4 [current_project]

# ── Set Verilog as default language ──────────────────────────
set_property target_language Verilog [current_project]

# ── Add all RTL source files ─────────────────────────────────
add_files -norecurse {
    rtl/pe_block.v
    rtl/pe_matrix.v
    rtl/pe_matrix_ctrl.v
    rtl/conv_maxpool_block.v
    rtl/mac_unit.v
    rtl/rom_module.v
    rtl/ram_module.v
    rtl/fc_control.v
    rtl/fsm_control.v
    rtl/uart.v
    rtl/lenet5_top.v
}

# ── Add simulation files ──────────────────────────────────────
add_files -fileset sim_1 -norecurse {
    sim/tb_lenet5_top.v
}

# ── Add constraints ───────────────────────────────────────────
add_files -fileset constrs_1 -norecurse {
    rtl/zedboard.xdc
}

# ── Set top module ────────────────────────────────────────────
set_property top lenet5_top [current_fileset]
set_property top tb_lenet5_top [get_filesets sim_1]

# ── Add hex weight files to project ──────────────────────────
# (Make sure hex_weights/ folder is in project root)
set_property include_dirs [list [pwd]/hex_weights] [current_fileset]

# ── Run synthesis ─────────────────────────────────────────────
# Uncomment to run automatically:
# launch_runs synth_1 -jobs 4
# wait_on_run synth_1

# ── Run implementation ────────────────────────────────────────
# launch_runs impl_1 -to_step write_bitstream -jobs 4
# wait_on_run impl_1

puts "=============================================="
puts " Project created: lenet5_zedboard"
puts " Next steps in Vivado GUI:"
puts "   1. Run Synthesis"
puts "   2. Run Implementation"
puts "   3. Generate Bitstream"
puts "   4. Program Device"
puts "=============================================="
