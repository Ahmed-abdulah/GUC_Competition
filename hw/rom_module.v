// ============================================================
//  rom_module.v  —  Distributed ROM for Trained Parameters
//  Paper reference : Mukhopadhyay et al., Fig. 6(a), Sec 3.2.4
//  Target          : ZedBoard XC7Z020, Vivado 2019.1
//
//  From paper Sec 3.2.4:
//    "A single input multiple output type RAM/ROM is used,
//     where five data samples are read simultaneously."
//    "ROM stores the fixed parameter values."
//    "distributed on-chip memory should be used for the cases
//     involving a fewer number of trained parameters that are
//     read in a parallel fashion." (Table 2 — saves ~30% BRAM)
//
//  This module implements a distributed ROM (Vivado infers
//  LUTRAM) with ONE read address and FIVE parallel outputs.
//  The five outputs provide 5 consecutive weight values,
//  matching the 5-column simultaneous read needed by the
//  PE matrix (one weight per PE column per cycle).
//
//  Usage:
//    Instantiate one per weight group:
//      conv1_w_rom  : 150   entries (6 filters × 5×5 × 1 ch)
//      conv2_w_rom  : 2400  entries (16 filters × 5×5 × 6 ch)
//      conv1_b_rom  : 6     entries
//      conv2_b_rom  : 16    entries
//      fc1_w_rom    : 30720 entries (120 neurons × 256 inputs)
//      fc1_b_rom    : 120   entries
//      fc2_w_rom    : 10080 entries (84 × 120)
//      fc2_b_rom    : 84    entries
//      fc3_w_rom    : 840   entries (10 × 84)
//      fc3_b_rom    : 10    entries
// ============================================================

module rom_module #(
    parameter DATA_W   = 8,
    parameter DEPTH    = 256,         // number of entries
    parameter INIT_FILE = "rom.hex"   // $readmemh init file
)(
    input  wire                      clk,
    input  wire                      en,           // read enable

    // ── Single address, 5 parallel outputs ───────────────
    // (paper: "single input multiple output")
    input  wire [$clog2(DEPTH)-1:0]  addr,         // base address

    output reg  signed [DATA_W-1:0]  dout0,        // addr + 0
    output reg  signed [DATA_W-1:0]  dout1,        // addr + 1
    output reg  signed [DATA_W-1:0]  dout2,        // addr + 2
    output reg  signed [DATA_W-1:0]  dout3,        // addr + 3
    output reg  signed [DATA_W-1:0]  dout4         // addr + 4
);

    // ── Distributed ROM (Vivado: (* rom_style = "distributed" *)) ─
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(INIT_FILE, mem);
    end

    // ── Synchronous read with enable ─────────────────────
    always @(posedge clk) begin
        if (en) begin
            dout0 <= mem[addr];
            // Guard against address out of range
            dout1 <= (addr+1 < DEPTH) ? mem[addr+1] : {DATA_W{1'b0}};
            dout2 <= (addr+2 < DEPTH) ? mem[addr+2] : {DATA_W{1'b0}};
            dout3 <= (addr+3 < DEPTH) ? mem[addr+3] : {DATA_W{1'b0}};
            dout4 <= (addr+4 < DEPTH) ? mem[addr+4] : {DATA_W{1'b0}};
        end
    end

endmodule
