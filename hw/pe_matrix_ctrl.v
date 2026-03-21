

module pe_matrix_ctrl #(
    parameter FILT = 5
)(
    input  wire            clk, rst,
    input  wire            start,
    output reg  [FILT-1:0] Clr_row,
    output reg  [FILT-1:0] Selp_row,
    output reg  [FILT-1:0] Seln_row,
    output reg  [FILT-1:0] col_en,
    output reg             out_valid,
    output reg             done
);
    localparam PERIOD  = 6;
    localparam COMPUTE = 5;
    localparam TOTAL   = FILT + FILT + PERIOD;

    reg [7:0] cycle_ctr;
    reg       running;

    // Compute local cycle for each row (inline, no array)
    // local_cycle[r] = (cycle_ctr - r) % PERIOD  when cycle_ctr >= r
    wire [2:0] lc0 = (cycle_ctr>=0) ? ((cycle_ctr-0)%PERIOD) : 3'd7;
    wire [2:0] lc1 = (cycle_ctr>=1) ? ((cycle_ctr-1)%PERIOD) : 3'd7;
    wire [2:0] lc2 = (cycle_ctr>=2) ? ((cycle_ctr-2)%PERIOD) : 3'd7;
    wire [2:0] lc3 = (cycle_ctr>=3) ? ((cycle_ctr-3)%PERIOD) : 3'd7;
    wire [2:0] lc4 = (cycle_ctr>=4) ? ((cycle_ctr-4)%PERIOD) : 3'd7;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            running<=0; cycle_ctr<=0;
            Clr_row<=0; Selp_row<=0; Seln_row<=0;
            col_en<=0; out_valid<=0; done<=0;
        end else begin
            out_valid <= 0;
            done      <= 0;

            if (start && !running) begin
                running   <= 1;
                cycle_ctr <= 0;
            end

            if (running) begin
                cycle_ctr <= cycle_ctr + 1;

                // Row 0
                Clr_row[0]  <= (lc0 == COMPUTE);
                Selp_row[0] <= 1'b0;  // row 0 never takes Prev
                Seln_row[0] <= (cycle_ctr >= 0) && (lc0 < COMPUTE);
                col_en[0]   <= (cycle_ctr >= 0) && (cycle_ctr < COMPUTE);

                // Row 1
                Clr_row[1]  <= (lc1 == COMPUTE);
                Selp_row[1] <= (cycle_ctr >= 1) && (lc1 < COMPUTE);
                Seln_row[1] <= (cycle_ctr >= 1) && (lc1 < COMPUTE);
                col_en[1]   <= (cycle_ctr >= 1) && (cycle_ctr < 1+COMPUTE);

                // Row 2
                Clr_row[2]  <= (lc2 == COMPUTE);
                Selp_row[2] <= (cycle_ctr >= 2) && (lc2 < COMPUTE);
                Seln_row[2] <= (cycle_ctr >= 2) && (lc2 < COMPUTE);
                col_en[2]   <= (cycle_ctr >= 2) && (cycle_ctr < 2+COMPUTE);

                // Row 3
                Clr_row[3]  <= (lc3 == COMPUTE);
                Selp_row[3] <= (cycle_ctr >= 3) && (lc3 < COMPUTE);
                Seln_row[3] <= (cycle_ctr >= 3) && (lc3 < COMPUTE);
                col_en[3]   <= (cycle_ctr >= 3) && (cycle_ctr < 3+COMPUTE);

                // Row 4 (last — out_valid tracks this row)
                Clr_row[4]  <= (lc4 == COMPUTE);
                Selp_row[4] <= (cycle_ctr >= 4) && (lc4 < COMPUTE);
                Seln_row[4] <= (cycle_ctr >= 4) && (lc4 < COMPUTE);
                col_en[4]   <= (cycle_ctr >= 4) && (cycle_ctr < 4+COMPUTE);

                // out_valid: last row producing valid output
                out_valid <= (cycle_ctr >= (2*FILT-1)) && (lc4 < COMPUTE);

                if (cycle_ctr == TOTAL - 1) begin
                    running   <= 0;
                    cycle_ctr <= 0;
                    done      <= 1;
                end
            end
        end
    end
endmodule
