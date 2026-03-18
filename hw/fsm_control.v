// ============================================================
//  fsm_control.v  —  Master CNN Control Unit
//  Paper reference : Mukhopadhyay et al., Fig. 10, Sec 3.2.5
//  Target          : ZedBoard XC7Z020, Vivado 2019.1
//
//  This FSM implements EXACTLY the state machine of paper
//  Fig. 10(a) with the EXACT signal names from the paper:
//
//  States (Fig. 10a):
//    0,1,2,3  : CONV1 → MAXPOOL1
//    4,5,6,7,8: CONV2 (6 batches) → MAXPOOL2 (6 batches)
//    9,10     : Summation of CONV2 batches (16 times)
//    11       : FC layer control
//
//  Timing signals (Fig. 10b):
//    rst_mat1c   : reset PE matrix for CONV1
//    rst_mat2c   : reset PE matrix for CONV2
//    rst_buff10  : reset buffer/output enable for CONV1
//    rst_buff2_i : reset buffer for CONV2 batch i
//    en_mxpl1    : enable MAXPOOL1
//    en_mxpl2_i  : enable MAXPOOL2 for batch i
//    en_contcnn  : enable FC layer control
//    en_add      : enable addition (sum CONV2 batch results)
//    i = 0,1,2,3,4,5 (6 batches)
// ============================================================

module fsm_control (
    input  wire        clk,
    input  wire        rst,

    // ── Trigger ───────────────────────────────────────────
    input  wire        img_ready,      // image received via UART

    // ── Done signals from each submodule ─────────────────
    input  wire        conv1_done,
    input  wire        mxpl1_done,
    input  wire        conv2_done,
    input  wire        mxpl2_done,
    input  wire        add_done,
    input  wire        fc_done,

    // ── Paper Fig. 10b timing/control signals ────────────
    output reg         rst_mat1c,      // CONV1 PE matrix reset (active LOW in paper → active HIGH here)
    output reg         rst_mat2c,      // CONV2 PE matrix reset
    output reg         rst_buff10,     // CONV1 output buffer enable
    output reg  [5:0]  rst_buff2_i,    // CONV2 batch buffer enable [5:0] for batches 0-5
    output reg         en_mxpl1,       // enable MAXPOOL1
    output reg  [5:0]  en_mxpl2_i,     // enable MAXPOOL2 per batch
    output reg         en_contcnn,     // enable FC control
    output reg         en_add,         // enable CONV2 batch summation

    // ── ROM enable signals ────────────────────────────────
    output reg         en_conv1_rom,   // enable CONV1 weight ROM
    output reg         en_conv2_rom,   // enable CONV2 weight ROM
    output reg         en_fc_rom,      // enable FC weight ROM

    // ── Filter/batch tracking ─────────────────────────────
    output reg  [3:0]  conv2_batch,    // current CONV2 batch 0..5
    output reg  [3:0]  add_count,      // addition counter 0..15

    // ── Inference complete ────────────────────────────────
    output reg         inference_done
);

    // ── States (Fig. 10a) ─────────────────────────────────
    localparam S0  = 4'd0;    // START / IDLE
    localparam S1  = 4'd1;    // Begin CONV1 (rst_mat1c low)
    localparam S2  = 4'd2;    // CONV1 running → write to MAXPOOL RAM (rst_buff10 low)
    localparam S3  = 4'd3;    // CONV1 done (rst_mat1c high)
    localparam S4  = 4'd4;    // All CONV1 data written (rst_buff10 high) → begin MAXPOOL1
    localparam S5  = 4'd5;    // MAXPOOL1 done → begin CONV2 batch (rst_mat2c low)
    localparam S6  = 4'd6;    // CONV2 running → write to MAXPOOL2 RAM (rst_buff2_i low)
    localparam S7  = 4'd7;    // CONV2 batch done (rst_mat2c high)
    localparam S8  = 4'd8;    // CONV2 data written → begin MAXPOOL2 (check batch count)
    localparam S9  = 4'd9;    // Begin addition (en_add high)
    localparam S10 = 4'd10;   // Check add counter (16 additions done?)
    localparam S11 = 4'd11;   // All done → enable FC (en_contcnn high)

    reg [3:0] state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state          <= S0;
            rst_mat1c      <= 1'b1;   // high = inactive (paper: low = active)
            rst_mat2c      <= 1'b1;
            rst_buff10     <= 1'b1;
            rst_buff2_i    <= 6'b111111;
            en_mxpl1       <= 1'b0;
            en_mxpl2_i     <= 6'b000000;
            en_contcnn     <= 1'b0;
            en_add         <= 1'b0;
            en_conv1_rom   <= 1'b0;
            en_conv2_rom   <= 1'b0;
            en_fc_rom      <= 1'b0;
            conv2_batch    <= 4'd0;
            add_count      <= 4'd0;
            inference_done <= 1'b0;
        end else begin
            // Default: deassert all start/enable pulses
            en_mxpl1       <= 1'b0;
            en_mxpl2_i     <= 6'b000000;
            en_contcnn     <= 1'b0;
            en_add         <= 1'b0;
            inference_done <= 1'b0;

            case (state)
                // ── S0: Wait for image ──────────────────────
                // Paper Fig. 10a: START state
                S0: begin
                    rst_mat1c    <= 1'b1;
                    rst_mat2c    <= 1'b1;
                    rst_buff10   <= 1'b1;
                    rst_buff2_i  <= 6'b111111;
                    en_conv1_rom <= 1'b0;
                    en_conv2_rom <= 1'b0;
                    en_fc_rom    <= 1'b0;
                    conv2_batch  <= 4'd0;
                    add_count    <= 4'd0;
                    if (img_ready) begin
                        en_conv1_rom <= 1'b1;
                        state        <= S1;
                    end
                end

                // ── S1: Begin CONV1 ─────────────────────────
                // Paper: "rst_mat1c is made low"
                S1: begin
                    rst_mat1c <= 1'b0;   // activate CONV1 PE matrix
                    state     <= S2;
                end

                // ── S2: CONV1 running, enable buffer write ──
                // Paper: "rst_buff10 signal is made low to enable
                //         writing in the RAM of the MAXPOOL module"
                S2: begin
                    rst_buff10 <= 1'b0;
                    if (conv1_done) state <= S3;
                end

                // ── S3: CONV1 done ───────────────────────────
                // Paper: "rst_mat1c is turned high as no more
                //         computation is required"
                S3: begin
                    rst_mat1c <= 1'b1;
                    state     <= S4;
                end

                // ── S4: Enable MAXPOOL1 ──────────────────────
                // Paper: "rst_buff10 is made high, all data has
                //         been written onto RAM and en_mxpl1 is
                //         made high to begin MAXPOOL"
                S4: begin
                    rst_buff10 <= 1'b1;
                    en_mxpl1   <= 1'b1;
                    if (mxpl1_done) state <= S5;
                end

                // ── S5: MAXPOOL1 done, begin CONV2 ──────────
                // Paper: "en_mxpl1 goes low, CONV2 operation
                //         starts, rst_mat2c is made low"
                // CONV2 runs in 6 batches (paper Sec 3.2.5)
                S5: begin
                    en_conv1_rom <= 1'b0;
                    en_conv2_rom <= 1'b1;
                    rst_mat2c    <= 1'b0;   // activate CONV2 PE matrix
                    state        <= S6;
                end

                // ── S6: CONV2 batch running, enable buffer ──
                // Paper: "rst_buff2_i signal is made low to
                //         enable writing in RAM of MAXPOOL module"
                S6: begin
                    rst_buff2_i[conv2_batch] <= 1'b0;
                    if (conv2_done) state <= S7;
                end

                // ── S7: CONV2 batch done ─────────────────────
                // Paper: "rst_mat2c is turned high as no more
                //         data need to be fetched"
                S7: begin
                    rst_mat2c <= 1'b1;
                    state     <= S8;
                end

                // ── S8: Enable MAXPOOL2 + check batch count ──
                // Paper: "rst_buff2_i made high, en_mxpl2_i made
                //         high to begin MAXPOOL. State 8 checks
                //         internal counter. If count > 6 → S9,
                //         else → S5"
                S8: begin
                    rst_buff2_i[conv2_batch] <= 1'b1;
                    en_mxpl2_i[conv2_batch]  <= 1'b1;
                    en_add                   <= 1'b1;   // sum results
                    if (mxpl2_done) begin
                        if (conv2_batch == 4'd5) begin
                            // All 6 batches done → summation
                            conv2_batch <= 4'd0;
                            state       <= S9;
                        end else begin
                            conv2_batch <= conv2_batch + 1;
                            rst_mat2c   <= 1'b0;   // next batch
                            state       <= S5;
                        end
                    end
                end

                // ── S9: Summation of CONV2 batch results ─────
                // Paper: "State 8 puts en_add signal high to
                //         enable addition. This addition is to
                //         take place 16 times."
                S9: begin
                    en_add    <= 1'b1;
                    add_count <= add_count + 1;
                    if (add_done) state <= S10;
                end

                // ── S10: Check add counter ───────────────────
                // Paper: "State 10 checks whether counter has
                //         reached 16, then S11, else S9"
                S10: begin
                    if (add_count == 4'd15) begin
                        add_count <= 4'd0;
                        state     <= S11;
                    end else begin
                        state <= S9;
                    end
                end

                // ── S11: Enable FC control ───────────────────
                // Paper: "control goes to FC-control block by
                //         making en_contcnn signal high"
                S11: begin
                    en_conv2_rom <= 1'b0;
                    en_fc_rom    <= 1'b1;
                    en_contcnn   <= 1'b1;
                    if (fc_done) begin
                        en_fc_rom      <= 1'b0;
                        inference_done <= 1'b1;
                        state          <= S0;
                    end
                end

                default: state <= S0;
            endcase
        end
    end

endmodule
