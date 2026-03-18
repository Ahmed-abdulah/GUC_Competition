// ============================================================
//  conv_maxpool_block.v  —  Convolution + MaxPooling Unit
//  Paper reference : Mukhopadhyay et al., Fig. 9(b), Sec 3.2.3
//  Target          : ZedBoard XC7Z020, Vivado 2019.1
//
//  Architecture (from paper, Fig. 9b):
//    "It consists of 16 PE matrices with associated tri-state
//     buffers. A 5×5×16 architecture is being made to model
//     the CONV operations of CNN. The sixteen 5×5 matrices
//     are processed in parallel. After the generation of the
//     values, they are written onto the RAM. The MAXPOOL unit
//     consists of 16 RAM blocks associated with 16 MAXPOOL
//     blocks each corresponding to a single 5×5 matrix."
//
//  For CONV1: 6 filters, 1 input channel
//    - 6 of the 16 PE matrices active (or run 1 at a time,
//      calling this block 6 times — paper runs 16 in parallel
//      but only needs 6 for CONV1; we parameterize N_FILTERS)
//  For CONV2: 16 filters, 6 input channels
//    - All 16 PE matrices active
//    - CONV2 runs in 6 batches of 16 (96 matrices total)
//
//  Memory (paper Sec 3.2.4):
//    - Single-input multiple-output ROM/RAM
//    - 5 data samples read simultaneously
//    - 6 RAM blocks for CONV1 stage
//    - 96 RAM blocks for CONV2 stage
//    - 1 RAM for output
// ============================================================

module conv_maxpool_block #(
    parameter DATA_W    = 8,
    parameter ACC_W     = 24,
    parameter FILT      = 5,        // 5×5 filter
    parameter N_FILT    = 16,       // number of parallel filters
    parameter IN_H      = 28,       // input height
    parameter IN_W      = 28,       // input width
    parameter IN_CH     = 1,        // input channels
    parameter FRAC_BITS = 6         // fixed-point fractional bits
)(
    input  wire                       clk,
    input  wire                       rst,

    // ── Control from master FSM (paper Fig. 10 signals) ───
    input  wire                       rst_matc,    // rst_mat1c / rst_mat2c
    input  wire                       rst_buff,    // rst_buff10 / rst_buff2i
    input  wire                       en_mxpl,     // en_mxpl1  / en_mxpl2i
    input  wire                       start,

    // ── Input feature map (from ROM/RAM) ──────────────────
    // 5 simultaneous reads (paper: "5 data samples read simultaneously")
    output reg  [15:0]                feat_rd_addr [0:4],
    input  wire signed [DATA_W-1:0]   feat_rd_data [0:4],

    // ── Weight ROM (distributed, one per filter) ──────────
    output reg  [11:0]                wt_rd_addr   [0:N_FILT-1],
    input  wire signed [DATA_W-1:0]   wt_rd_data   [0:N_FILT-1],

    // ── Bias ROM ──────────────────────────────────────────
    output reg  [3:0]                 bias_addr    [0:N_FILT-1],
    input  wire signed [DATA_W-1:0]   bias_data    [0:N_FILT-1],

    // ── Output RAM (one port per filter for parallel write) ─
    output reg  [11:0]                feat_wr_addr [0:N_FILT-1],
    output reg  signed [DATA_W-1:0]   feat_wr_data [0:N_FILT-1],
    output reg  [N_FILT-1:0]         feat_wr_en,

    // ── MaxPool output RAM ────────────────────────────────
    output reg  [7:0]                 mxpl_wr_addr [0:N_FILT-1],
    output reg  signed [DATA_W-1:0]   mxpl_wr_data [0:N_FILT-1],
    output reg  [N_FILT-1:0]         mxpl_wr_en,

    // ── Status ────────────────────────────────────────────
    output reg                        conv_done,   // CONV complete
    output reg                        mxpl_done    // MAXPOOL complete
);

    // Output map dimensions
    localparam OUT_H = IN_H - FILT + 1;   // CONV1: 28-5+1=24
    localparam OUT_W = IN_W - FILT + 1;
    localparam POOL_H = OUT_H / 2;        // after MAXPOOL: 12
    localparam POOL_W = OUT_W / 2;

    // ── FSM ──────────────────────────────────────────────
    localparam S_IDLE      = 4'd0;
    localparam S_CONV_INIT = 4'd1;
    localparam S_CONV_LOAD = 4'd2;   // load 5 pixel columns
    localparam S_CONV_WAIT = 4'd3;   // wait for PE matrix
    localparam S_CONV_OUT  = 4'd4;   // write output to RAM
    localparam S_CONV_NEXT = 4'd5;   // next output position
    localparam S_POOL_INIT = 4'd6;
    localparam S_POOL_READ = 4'd7;
    localparam S_POOL_WAIT = 4'd8;
    localparam S_POOL_WRITE= 4'd9;
    localparam S_POOL_NEXT = 4'd10;
    localparam S_DONE      = 4'd11;

    reg [3:0]  state;
    reg [4:0]  out_row, out_col;      // output pixel position
    reg [4:0]  pool_row, pool_col;    // pooling position
    reg [3:0]  filt_idx;              // current filter 0..N_FILT-1
    reg [3:0]  wait_ctr;

    // ── PE accumulator registers (one per filter) ─────────
    reg signed [ACC_W-1:0] acc [0:N_FILT-1];
    reg [4:0]               mac_idx;    // current weight position

    // ── ReLU function ─────────────────────────────────────
    function signed [DATA_W-1:0] relu8;
        input signed [ACC_W-1:0] x;
        begin
            if (x[ACC_W-1])                          // negative
                relu8 = 8'sd0;
            else if (|x[ACC_W-2:DATA_W+FRAC_BITS-1]) // overflow
                relu8 = 8'sd127;
            else
                relu8 = x[DATA_W+FRAC_BITS-2:FRAC_BITS];
        end
    endfunction

    // ── MaxPool registers ─────────────────────────────────
    reg signed [DATA_W-1:0] pool_max  [0:N_FILT-1];
    reg signed [DATA_W-1:0] pool_p0   [0:N_FILT-1];
    reg signed [DATA_W-1:0] pool_p1   [0:N_FILT-1];
    reg signed [DATA_W-1:0] pool_p2   [0:N_FILT-1];
    reg [1:0]               pool_step;

    integer i;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state      <= S_IDLE;
            conv_done  <= 0;
            mxpl_done  <= 0;
            out_row    <= 0;
            out_col    <= 0;
            pool_row   <= 0;
            pool_col   <= 0;
            filt_idx   <= 0;
            mac_idx    <= 0;
            wait_ctr   <= 0;
            feat_wr_en <= 0;
            mxpl_wr_en <= 0;
            for (i = 0; i < N_FILT; i = i+1) begin
                acc[i]       <= 0;
                pool_max[i]  <= -8'sd128;
                feat_wr_en[i]<= 1'b0;
                mxpl_wr_en[i]<= 1'b0;
            end
        end else begin
            conv_done  <= 0;
            mxpl_done  <= 0;
            feat_wr_en <= 0;
            mxpl_wr_en <= 0;

            case (state)
                // ── Wait for master FSM start signal ────────
                S_IDLE: begin
                    if (start && !rst_matc) begin
                        out_row  <= 0;
                        out_col  <= 0;
                        mac_idx  <= 0;
                        // Load bias as initial accumulator value
                        for (i = 0; i < N_FILT; i = i+1)
                            bias_addr[i] <= i[3:0];
                        state <= S_CONV_INIT;
                    end
                end

                // ── Load bias into all filter accumulators ───
                S_CONV_INIT: begin
                    for (i = 0; i < N_FILT; i = i+1) begin
                        acc[i] <= {{(ACC_W-DATA_W){bias_data[i][DATA_W-1]}},
                                    bias_data[i]} <<< FRAC_BITS;
                    end
                    mac_idx <= 0;
                    state   <= S_CONV_LOAD;
                end

                // ── Set up 5 parallel feature + weight reads ─
                // Paper: "5 data samples read simultaneously"
                S_CONV_LOAD: begin
                    // Read 5 pixels from current filter row position
                    // (one pixel per column of the 5×5 filter)
                    for (i = 0; i < 5; i = i+1) begin
                        feat_rd_addr[i] <=
                            (out_row + mac_idx/FILT) * IN_W +
                            (out_col + i);
                    end
                    // Read weights for all filters simultaneously
                    for (i = 0; i < N_FILT; i = i+1) begin
                        wt_rd_addr[i] <= mac_idx;
                    end
                    state <= S_CONV_WAIT;
                end

                // ── Wait one cycle for distributed ROM read ──
                S_CONV_WAIT: begin
                    state <= S_CONV_OUT;
                end

                // ── Accumulate: multiply all 5 pixels × weight
                // Each filter uses the same pixel, different weight
                S_CONV_OUT: begin
                    for (i = 0; i < N_FILT; i = i+1) begin
                        // Sum 5 multiplications (one per column)
                        acc[i] <= acc[i] +
                            $signed(wt_rd_data[i]) * feat_rd_data[mac_idx % FILT];
                    end

                    if (mac_idx == FILT*FILT*IN_CH - 1) begin
                        // All 25 (or 150 for CONV2) MACs done
                        // Apply ReLU and write to output RAM
                        for (i = 0; i < N_FILT; i = i+1) begin
                            feat_wr_data[i] <= relu8(acc[i]);
                            feat_wr_addr[i] <= i * OUT_H * OUT_W +
                                               out_row * OUT_W + out_col;
                            feat_wr_en[i]   <= 1'b1;
                        end
                        mac_idx <= 0;
                        state   <= S_CONV_NEXT;
                    end else begin
                        mac_idx <= mac_idx + 1;
                        state   <= S_CONV_LOAD;
                    end
                end

                // ── Move to next output pixel ────────────────
                S_CONV_NEXT: begin
                    if (out_col == OUT_W - 1) begin
                        out_col <= 0;
                        if (out_row == OUT_H - 1) begin
                            out_row   <= 0;
                            conv_done <= 1;
                            if (rst_matc) begin
                                state <= S_IDLE;
                            end else if (en_mxpl) begin
                                // Begin MAXPOOL immediately
                                pool_row  <= 0;
                                pool_col  <= 0;
                                pool_step <= 0;
                                for (i=0;i<N_FILT;i=i+1)
                                    pool_max[i] <= -8'sd128;
                                state <= S_POOL_INIT;
                            end else begin
                                state <= S_IDLE;
                            end
                        end else begin
                            out_row <= out_row + 1;
                            state   <= S_CONV_INIT;
                        end
                    end else begin
                        out_col <= out_col + 1;
                        state   <= S_CONV_INIT;
                    end
                end

                // ── MAXPOOL: 2×2 window ──────────────────────
                // Paper: "MAXPOOL unit consists of 16 RAM blocks
                //         associated with 16 MAXPOOL blocks"
                S_POOL_INIT: begin
                    if (en_mxpl) begin
                        for (i = 0; i < N_FILT; i = i+1) begin
                            pool_max[i] <= -8'sd128;
                            feat_rd_addr[0] <=
                                i * OUT_H * OUT_W +
                                (2*pool_row)   * OUT_W + (2*pool_col);
                            feat_rd_addr[1] <=
                                i * OUT_H * OUT_W +
                                (2*pool_row)   * OUT_W + (2*pool_col+1);
                            feat_rd_addr[2] <=
                                i * OUT_H * OUT_W +
                                (2*pool_row+1) * OUT_W + (2*pool_col);
                            feat_rd_addr[3] <=
                                i * OUT_H * OUT_W +
                                (2*pool_row+1) * OUT_W + (2*pool_col+1);
                        end
                        pool_step <= 0;
                        state     <= S_POOL_READ;
                    end
                end

                S_POOL_READ: begin
                    state <= S_POOL_WAIT;
                end

                S_POOL_WAIT: begin
                    // Compute max of 4 values per filter
                    for (i = 0; i < N_FILT; i = i+1) begin
                        pool_p0[i] <= feat_rd_data[0];
                        pool_p1[i] <= feat_rd_data[1];
                        pool_p2[i] <= feat_rd_data[2];
                        // feat_rd_data[3] = 4th pixel
                        pool_max[i] <=
                            (feat_rd_data[0] > feat_rd_data[1] ?
                                feat_rd_data[0] : feat_rd_data[1]) >
                            (feat_rd_data[2] > feat_rd_data[3] ?
                                feat_rd_data[2] : feat_rd_data[3]) ?
                            (feat_rd_data[0] > feat_rd_data[1] ?
                                feat_rd_data[0] : feat_rd_data[1]) :
                            (feat_rd_data[2] > feat_rd_data[3] ?
                                feat_rd_data[2] : feat_rd_data[3]);
                    end
                    state <= S_POOL_WRITE;
                end

                S_POOL_WRITE: begin
                    for (i = 0; i < N_FILT; i = i+1) begin
                        mxpl_wr_data[i] <= pool_max[i];
                        mxpl_wr_addr[i] <= i * POOL_H * POOL_W +
                                           pool_row * POOL_W + pool_col;
                        mxpl_wr_en[i]   <= 1'b1;
                    end
                    state <= S_POOL_NEXT;
                end

                S_POOL_NEXT: begin
                    if (pool_col == POOL_W - 1) begin
                        pool_col <= 0;
                        if (pool_row == POOL_H - 1) begin
                            pool_row  <= 0;
                            mxpl_done <= 1;
                            state     <= S_DONE;
                        end else begin
                            pool_row <= pool_row + 1;
                            state    <= S_POOL_INIT;
                        end
                    end else begin
                        pool_col <= pool_col + 1;
                        state    <= S_POOL_INIT;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
