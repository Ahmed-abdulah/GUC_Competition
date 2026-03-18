// ============================================================
//  ram_module.v  —  Dual-Port RAM for Feature Maps
//  Paper reference : Mukhopadhyay et al., Fig. 6(b), Sec 3.1.2
//  Target          : ZedBoard XC7Z020, Vivado 2019.1
//
//  From paper Fig. 6(b) and Sec 3.1.2:
//    "A random-access memory (RAM) is used for storing the
//     input FV and also the result obtained by the activation
//     unit, which serves as the input to the subsequent layer."
//    Ports: wr_en, rd_en, addr_wr, addr_rd, clk_wr, clk_rd
//           din, dout
//
//  This is a true dual-port RAM:
//    - Port A: write port (from conv/fc outputs)
//    - Port B: read port  (to next layer inputs)
//  Both ports share one clock (synchronous design per paper).
//
//  Memory map (103 RAM blocks in paper):
//    Block 0     : input image       784  bytes
//    Blocks 1-6  : CONV1 output      24×24×6  = 3456 bytes
//    Blocks 7-12 : POOL1 output      12×12×6  = 864  bytes
//    Blocks 13-108: CONV2 output     8×8×16×6 = 6144 bytes
//    Blocks 109-124: POOL2 output    4×4×16   = 256  bytes
//    Block 125   : FC1 output        120 bytes
//    Block 126   : FC2 output        84  bytes
//    Block 127   : FC3 output        10  bytes  (flattened output RAM)
// ============================================================

module ram_module #(
    parameter DATA_W = 8,
    parameter DEPTH  = 1024       // entries per RAM block
)(
    input  wire                      clk,

    // ── Write port (Port A) ──────────────────────────────
    input  wire                      wr_en,
    input  wire [$clog2(DEPTH)-1:0]  addr_wr,
    input  wire signed [DATA_W-1:0]  din,

    // ── Read port (Port B) ───────────────────────────────
    input  wire                      rd_en,
    input  wire [$clog2(DEPTH)-1:0]  addr_rd,
    output reg  signed [DATA_W-1:0]  dout
);

    // Vivado infers Block RAM for depths >= 512, LUTRAM for smaller
    reg signed [DATA_W-1:0] mem [0:DEPTH-1];

    // ── Write (synchronous) ──────────────────────────────
    always @(posedge clk) begin
        if (wr_en)
            mem[addr_wr] <= din;
    end

    // ── Read (synchronous) ───────────────────────────────
    always @(posedge clk) begin
        if (rd_en)
            dout <= mem[addr_rd];
    end

endmodule
