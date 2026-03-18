// ============================================================
//  pe_matrix.v  —  5×5 Systolic Array of PE Blocks
//  Paper reference : Mukhopadhyay et al., Fig. 9(c)
//  Target          : ZedBoard XC7Z020, Vivado 2019.1
//
//  Architecture (Fig. 9(c)):
//    25 PE blocks arranged in 5 rows × 5 columns.
//    Data flows diagonally through the matrix.
//    - Columns receive input pixels (a_ij) at staggered times
//    - Rows receive filter weights (b_ij) at staggered times
//    - Output accumulates row-by-row through the 5 PEs in each column
//
//  Timing (paper Section 3.2.2):
//    - 1c = 1 clock cycle
//    - 1d = 6 clock cycles (5 compute + 1 reset)
//    - Each PE resets (Clr) every 5 clock cycles
//    - Different PE blocks start and end at different times
//    - Output appears at the last PE of each column
//
//  Connections:
//    - pe[row][col].In   ← input pixel for that column (skewed by col)
//    - pe[row][col].W    ← weight for that position
//    - pe[row][col].Prev ← pe[row-1][col].Out  (vertical chain)
//    - pe[row][col].Out  → pe[row+1][col].In or final output
//
//  Data layout:
//    Inputs  a[row][col] are presented with 1-cycle skew per column
//    Weights b[row][col] are presented with 1-cycle skew per row
//    C[i][j] = Σ_k  a[i][k] × b[k][j]  (convolution slice)
// ============================================================

module pe_matrix #(
    parameter DATA_W  = 8,
    parameter ACC_W   = 24,
    parameter FILT    = 5       // filter dimension → 5×5 = 25 PEs
)(
    input  wire                        clk,
    input  wire                        rst,

    // ── Input pixels: one per column, staggered externally ─
    input  wire signed [DATA_W-1:0]    a [0:FILT-1],   // a[col]

    // ── Filter weights: one per (row,col) ──────────────────
    input  wire signed [DATA_W-1:0]    b [0:FILT-1][0:FILT-1], // b[row][col]

    // ── Control: Clr resets every 5 cycles per PE ──────────
    // Supplied externally by PE matrix control block
    input  wire [FILT-1:0]             Clr_row,   // per-row clear
    input  wire [FILT-1:0]             Selp_row,  // per-row Selp
    input  wire [FILT-1:0]             Seln_row,  // per-row Seln

    // ── Output: result of each column (from last row PE) ───
    output wire signed [ACC_W-1:0]     C [0:FILT-1],  // C[col]

    // ── Valid: output is ready (last row has completed) ────
    output reg                         out_valid
);

    // ── Intermediate wires between PE rows ───────────────────
    // prev[row][col] = output of pe[row-1][col]
    wire signed [ACC_W-1:0] prev [0:FILT][0:FILT-1];

    // Row 0 receives zero as previous (start of chain)
    genvar c0;
    generate
        for (c0 = 0; c0 < FILT; c0 = c0+1) begin : gen_prev0
            assign prev[0][c0] = {ACC_W{1'b0}};
        end
    endgenerate

    // ── Instantiate 5×5 = 25 PE blocks ───────────────────────
    genvar r, c;
    generate
        for (r = 0; r < FILT; r = r+1) begin : gen_row
            for (c = 0; c < FILT; c = c+1) begin : gen_col
                pe_block #(
                    .DATA_W(DATA_W),
                    .ACC_W (ACC_W)
                ) u_pe (
                    .clk  (clk),
                    .rst  (rst),
                    .In   (a[c]),              // column input
                    .W    (b[r][c]),           // weight at (row,col)
                    .Prev (prev[r][c]),        // from row above
                    .Clr  (Clr_row[r]),        // row-level clear
                    .Selp (Selp_row[r]),       // row-level Selp
                    .Seln (Seln_row[r]),       // row-level Seln
                    .Out  (prev[r+1][c])       // feeds row below
                );
            end
        end
    endgenerate

    // ── Final row outputs = convolution result ────────────────
    genvar oc;
    generate
        for (oc = 0; oc < FILT; oc = oc+1) begin : gen_out
            assign C[oc] = prev[FILT][oc];
        end
    endgenerate

    // ── out_valid: asserted when last row has valid data ─────
    // Simplistic: assert after FILT×2 cycles from start
    // (detailed timing handled by pe_matrix_ctrl)
    reg [3:0] valid_ctr;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_ctr <= 0;
            out_valid <= 0;
        end else begin
            if (Seln_row[FILT-1]) begin
                out_valid <= 1;
            end else begin
                out_valid <= 0;
            end
        end
    end

endmodule
