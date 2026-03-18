// ============================================================
//  pe_matrix_ctrl.v  —  PE Matrix Control Block
//  Paper reference : Mukhopadhyay et al., Section 3.2.5
//                    "PE matrix control"
//  Target          : ZedBoard XC7Z020, Vivado 2019.1
//
//  Responsibilities (from paper):
//    "PE matrix control will generate signals to regulate the
//     data flow into the matrix and also to produce the proper
//     output. Each PE block needs to be reset after 5 clock
//     cycles, while different PE blocks start and end at
//     different times."
//
//  Timing (paper Fig. 9c):
//    c = 1 clock cycle
//    d = 6 clock cycles (5 compute + 1 Clr)
//
//    PE[row][col] starts at time: (row + col) × 1c
//    PE[row][col] outputs at  : (row + col + FILT) × 1c
//
//    Clr_row[r]  asserted every 6th cycle for row r
//    Selp_row[r] = 1 when row r > 0 and prev row has valid data
//    Seln_row[r] = 1 during the 5 compute cycles
// ============================================================

module pe_matrix_ctrl #(
    parameter FILT = 5       // 5×5 filter
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              start,      // start processing

    // ── Control outputs to pe_matrix ─────────────────────
    output reg  [FILT-1:0]  Clr_row,    // clear per row
    output reg  [FILT-1:0]  Selp_row,   // Selp per row
    output reg  [FILT-1:0]  Seln_row,   // Seln per row

    // ── Column input enable (stagger data input) ──────────
    output reg  [FILT-1:0]  col_en,     // enable column input

    // ── Output valid: last PE column has result ───────────
    output reg              out_valid,
    output reg              done         // all output pixels done
);

    // Cycle counter: counts clock cycles since start
    reg [7:0] cycle_ctr;
    reg       running;

    // Period = 6 cycles (5 compute + 1 reset)
    localparam PERIOD = 6;
    localparam COMPUTE = 5;

    // Local cycle within current period for each row
    // Row r starts at cycle r (staggered by 1 cycle each row)
    wire [2:0] local_cycle [0:FILT-1];
    genvar r;
    generate
        for (r = 0; r < FILT; r = r+1) begin : gen_lc
            assign local_cycle[r] = (cycle_ctr >= r) ?
                                    ((cycle_ctr - r) % PERIOD) : 3'd7;
        end
    endgenerate

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            cycle_ctr  <= 0;
            running    <= 0;
            Clr_row    <= {FILT{1'b0}};
            Selp_row   <= {FILT{1'b0}};
            Seln_row   <= {FILT{1'b0}};
            col_en     <= {FILT{1'b0}};
            out_valid  <= 0;
            done       <= 0;
        end else begin
            out_valid <= 0;
            done      <= 0;

            if (start && !running) begin
                running   <= 1;
                cycle_ctr <= 0;
            end

            if (running) begin
                cycle_ctr <= cycle_ctr + 1;

                // ── Generate Clr, Selp, Seln per row ─────────
                begin : ctrl_gen
                    integer i;
                    for (i = 0; i < FILT; i = i+1) begin
                        // Clr on cycle 5 of each period (reset cycle)
                        Clr_row[i]  <= (local_cycle[i] == COMPUTE);
                        // Selp: row 0 gets no prev; rows 1-4 get prev
                        Selp_row[i] <= (i > 0) &&
                                       (cycle_ctr >= i) &&
                                       (local_cycle[i] < COMPUTE);
                        // Seln: pass output during compute cycles
                        Seln_row[i] <= (cycle_ctr >= i) &&
                                       (local_cycle[i] < COMPUTE);
                    end
                end

                // ── Stagger column input enables ──────────────
                begin : col_gen
                    integer j;
                    for (j = 0; j < FILT; j = j+1) begin
                        col_en[j] <= (cycle_ctr >= j) &&
                                     (cycle_ctr < j + COMPUTE);
                    end
                end

                // ── Output valid: last row last column ready ──
                // Output ready when cycle >= 2*FILT-1
                if (cycle_ctr >= (2*FILT - 1) &&
                    local_cycle[FILT-1] < COMPUTE) begin
                    out_valid <= 1;
                end

                // ── Done after one full pass ───────────────────
                // Enough cycles for all PEs: FILT + FILT-1 + PERIOD
                if (cycle_ctr == (FILT + FILT + PERIOD - 1)) begin
                    running   <= 0;
                    cycle_ctr <= 0;
                    done      <= 1;
                end
            end
        end
    end

endmodule
