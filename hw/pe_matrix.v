

module pe_matrix #(
    parameter DATA_W = 8,
    parameter ACC_W  = 24,
    parameter FILT   = 5
)(
    input  wire                          clk,
    input  wire                          rst,
    // Flat packed inputs (unpack inside)
    input  wire [FILT*DATA_W-1:0]        a_flat,     // a[col] = a_flat[col*DATA_W+:DATA_W]
    input  wire [FILT*FILT*DATA_W-1:0]   b_flat,     // b[r][c]= b_flat[(r*FILT+c)*DATA_W+:DATA_W]
    // Per-row control signals
    input  wire [FILT-1:0]               Clr_row,
    input  wire [FILT-1:0]               Selp_row,
    input  wire [FILT-1:0]               Seln_row,
    // Flat packed output
    output wire [FILT*ACC_W-1:0]         C_flat,     // C[col]= C_flat[col*ACC_W+:ACC_W]
    output reg                           out_valid
);
    // ── Vertical chain wires: prev[r][c] ──────────────────
    // prev[0][c] = 0 (top of column)
    // prev[r+1][c] = pe[r][c].Out
    wire signed [ACC_W-1:0] prev [0:FILT][0:FILT-1];

    genvar ci;
    generate
        for (ci = 0; ci < FILT; ci = ci+1) begin : gen_prev0
            assign prev[0][ci] = {ACC_W{1'b0}};
        end
    endgenerate

    // ── 5×5 PE block array ────────────────────────────────
    genvar ri, cj;
    generate
        for (ri = 0; ri < FILT; ri = ri+1) begin : gen_row
            for (cj = 0; cj < FILT; cj = cj+1) begin : gen_col
                pe_block #(
                    .DATA_W(DATA_W),
                    .ACC_W (ACC_W)
                ) u_pe (
                    .clk (clk),
                    .rst (rst),
                    // a[col] from flat pack
                    .In  (a_flat[(cj*DATA_W) +: DATA_W]),
                    // b[row][col] from flat pack (row-major)
                    .W   (b_flat[((ri*FILT+cj)*DATA_W) +: DATA_W]),
                    // vertical chain
                    .Prev(prev[ri][cj]),
                    // per-row control
                    .Clr (Clr_row[ri]),
                    .Selp(Selp_row[ri]),
                    .Seln(Seln_row[ri]),
                    // feeds next row
                    .Out (prev[ri+1][cj])
                );
            end
        end
    endgenerate

    // ── Output: last row results ──────────────────────────
    genvar oc;
    generate
        for (oc = 0; oc < FILT; oc = oc+1) begin : gen_out
            assign C_flat[(oc*ACC_W) +: ACC_W] = prev[FILT][oc];
        end
    endgenerate

    // ── out_valid: driven by pe_matrix_ctrl externally ────
    always @(posedge clk) begin
        if (rst) out_valid <= 1'b0;
        else     out_valid <= Seln_row[FILT-1];
    end

endmodule
