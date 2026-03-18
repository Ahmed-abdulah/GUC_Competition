// ============================================================
//  pe_block.v  —  Processing Element (PE)
//  Paper reference : Mukhopadhyay et al., Fig. 9(a)
//  Target          : ZedBoard XC7Z020, Vivado 2019.1
//
//  Structure (Fig. 9a):
//    - One multiplier      : In × W
//    - One 3-input adder   : mult_out + Prev + add_in
//    - One output register : D-FF with CE and CLR
//    - Two MUXes           : controlled by Selp and Seln
//
//  Control signals (from paper):
//    Clr  : clear internal register (synchronous)
//    Selp : 1 = receive partial sum from previous PE (Prev input)
//           0 = Prev input is 0 (start of chain)
//    Seln : 1 = pass output to next PE downstream
//           0 = hold (tristate / not forwarding)
//
//  Data flow (paper Fig. 9c timing):
//    - 5 clock cycles for computation (one per filter row)
//    - 1 clock cycle for reset (Clr)
//    - 6 cycles total = 1 "d" cycle in paper notation
// ============================================================

module pe_block #(
    parameter DATA_W = 8,    // input / weight bit-width (fixed-point)
    parameter ACC_W  = 24    // accumulator width — prevents overflow
                             // ACC_W >= 2*DATA_W + ceil(log2(25)) = 21 → use 24
)(
    input  wire                       clk,
    input  wire                       rst,

    // ── Data inputs ──────────────────────────────────────
    input  wire signed [DATA_W-1:0]   In,       // pixel / feature input
    input  wire signed [DATA_W-1:0]   W,        // filter weight
    input  wire signed [ACC_W-1:0]    Prev,     // partial sum from prev PE

    // ── Control signals (paper Fig. 9a) ──────────────────
    input  wire                       Clr,      // synchronous clear
    input  wire                       Selp,     // MUX: select Prev input
    input  wire                       Seln,     // MUX: forward output

    // ── Outputs ──────────────────────────────────────────
    output wire signed [ACC_W-1:0]    Out       // to next PE in chain
);

    // ── Multiply stage ────────────────────────────────────
    wire signed [2*DATA_W-1:0] mult_result;
    assign mult_result = In * W;   // 8×8 = 16-bit product

    // Sign-extend to ACC_W
    wire signed [ACC_W-1:0] mult_ext;
    assign mult_ext = {{(ACC_W-2*DATA_W){mult_result[2*DATA_W-1]}},
                        mult_result};

    // ── MUX: select Prev or zero (Selp) ──────────────────
    wire signed [ACC_W-1:0] prev_mux;
    assign prev_mux = Selp ? Prev : {ACC_W{1'b0}};

    // ── 3-input adder ─────────────────────────────────────
    wire signed [ACC_W-1:0] add_result;
    assign add_result = mult_ext + prev_mux;

    // ── Output register (D-FF with CE and synchronous CLR) ─
    reg signed [ACC_W-1:0] reg_out;
    always @(posedge clk) begin
        if (rst || Clr)
            reg_out <= {ACC_W{1'b0}};
        else
            reg_out <= add_result;
    end

    // ── MUX: forward to next PE or hold (Seln) ────────────
    // In the paper, Seln=1 passes output downstream.
    // We use a wire (not tri-state) for FPGA compatibility;
    // the pe_matrix wires PEs in a chain and gates with Seln.
    assign Out = Seln ? reg_out : {ACC_W{1'b0}};

endmodule
