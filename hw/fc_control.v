// ============================================================
//  fc_control.v  —  FC Layer Control Unit
//  Paper reference : Mukhopadhyay et al., Sec 3.1.3, Fig. 7
//  Target          : ZedBoard XC7Z020, Vivado 2019.1
//
//  Implements FC layer FSM from Fig. 7 EXACTLY:
//  States 2,3,4,5 for each hidden layer then output layer.
//
//  "State 5 is a decider state, which checks whether all
//   neurons in the hidden layer are processed. If all neurons
//   are covered, then the controller moves to the next state,
//   else it goes back to state 2."
//
//  States 2,3,4,5  → FC1  (256→120 with ReLU)
//  States 6,7,8,9  → FC2  (120→84  with ReLU)
//  States 10,11,12,13 → FC3 (84→10  no ReLU)
//  State 14 → STOP
// ============================================================

module fc_control #(
    parameter DATA_W    = 8,
    parameter ACC_W     = 24,
    parameter N_FC1_IN  = 256,
    parameter N_FC1_OUT = 120,
    parameter N_FC2_IN  = 120,
    parameter N_FC2_OUT = 84,
    parameter N_FC3_IN  = 84,
    parameter N_FC3_OUT = 10,
    parameter FRAC_BITS = 6
)(
    input  wire                       clk,
    input  wire                       rst,
    input  wire                       en_contcnn,    // from master FSM S11

    // ── Feature vector RAM (input) ────────────────────────
    output reg  [15:0]                fv_rd_addr,
    input  wire signed [DATA_W-1:0]   fv_rd_data,
    input  wire                       fv_rd_en,

    // ── Weight ROM ────────────────────────────────────────
    output reg  [17:0]                wt_rd_addr,
    input  wire signed [DATA_W-1:0]   wt_rd_data,

    // ── Bias ROM ──────────────────────────────────────────
    output reg  [9:0]                 bias_rd_addr,
    input  wire signed [DATA_W-1:0]   bias_rd_data,

    // ── Output RAM (write) ────────────────────────────────
    output reg  [9:0]                 out_wr_addr,
    output reg  signed [DATA_W-1:0]   out_wr_data,
    output reg                        out_wr_en,
    output reg                        fv_wr_en,
    output reg  [15:0]                fv_wr_addr,
    output reg  signed [DATA_W-1:0]   fv_wr_data,

    // ── MAC control signals (paper Fig. 5a) ───────────────
    output reg                        bias_en,
    output reg                        load_acc,
    input  wire signed [DATA_W-1:0]   act_out,       // from mac_unit

    // ── Status ────────────────────────────────────────────
    output reg                        fc_done
);

    // ── FSM States (matching paper Fig. 7) ────────────────
    // State 0,1: global reset (handled by master FSM)
    localparam S_IDLE  = 4'd0;
    localparam S2      = 4'd2;   // load bias
    localparam S3      = 4'd3;   // MAC loop (N_INP cycles)
    localparam S4      = 4'd4;   // ReLU + store result
    localparam S5      = 4'd5;   // check if all neurons done
    localparam S6      = 4'd6;   // FC2: load bias
    localparam S7      = 4'd7;   // FC2: MAC loop
    localparam S8      = 4'd8;   // FC2: ReLU + store
    localparam S9      = 4'd9;   // FC2: check neurons
    localparam S10     = 4'd10;  // FC3: load bias
    localparam S11     = 4'd11;  // FC3: MAC loop
    localparam S12     = 4'd12;  // FC3: store (no ReLU)
    localparam S13     = 4'd13;  // FC3: check neurons
    localparam S14     = 4'd14;  // STOP

    reg [3:0]  state;
    reg [15:0] input_ctr;    // input neuron counter
    reg [9:0]  neuron_ctr;   // output neuron counter
    reg [9:0]  n_in_cur;     // current layer input count
    reg [9:0]  n_out_cur;    // current layer output count
    reg [17:0] wt_base;      // weight ROM base address for layer
    reg [9:0]  bias_base;    // bias ROM base address for layer
    reg [15:0] fv_in_base;   // input feature RAM base address
    reg [15:0] fv_out_base;  // output feature RAM base address
    reg        use_relu;     // 0 for FC3

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state        <= S_IDLE;
            fc_done      <= 0;
            bias_en      <= 0;
            load_acc     <= 0;
            out_wr_en    <= 0;
            fv_wr_en     <= 0;
            fv_rd_addr   <= 0;
            wt_rd_addr   <= 0;
            bias_rd_addr <= 0;
            out_wr_addr  <= 0;
            fv_wr_addr   <= 0;
            input_ctr    <= 0;
            neuron_ctr   <= 0;
        end else begin
            bias_en   <= 0;
            load_acc  <= 0;
            out_wr_en <= 0;
            fv_wr_en  <= 0;
            fc_done   <= 0;

            case (state)
                S_IDLE: begin
                    if (en_contcnn) begin
                        // Start FC1: 256 inputs → 120 outputs
                        n_in_cur    <= N_FC1_IN;
                        n_out_cur   <= N_FC1_OUT;
                        wt_base     <= 18'd0;
                        bias_base   <= 10'd0;
                        fv_in_base  <= 16'd0;     // pool2 output
                        fv_out_base <= 16'd256;   // FC1 output area
                        use_relu    <= 1;
                        neuron_ctr  <= 0;
                        state       <= S2;
                    end
                end

                // ── FC1 States (States 2-5 in paper Fig. 7) ─
                // State 2: load bias for current neuron
                S2: begin
                    bias_rd_addr <= bias_base + neuron_ctr[9:0];
                    state        <= S2 + 1;   // → S3
                end

                // State 3: MAC loop over all inputs
                S3: begin
                    // First cycle: load bias into accumulator
                    if (input_ctr == 0) begin
                        bias_en <= 1;
                    end
                    // Set read addresses
                    fv_rd_addr <= fv_in_base + input_ctr;
                    wt_rd_addr <= wt_base + neuron_ctr * n_in_cur + input_ctr;
                    load_acc   <= 1;
                    input_ctr  <= input_ctr + 1;
                    if (input_ctr == n_in_cur - 1) begin
                        input_ctr <= 0;
                        state     <= S4;
                    end
                end

                // State 4: apply activation, write to RAM
                S4: begin
                    // act_out comes from mac_unit (ReLU already applied)
                    fv_wr_addr <= fv_out_base + neuron_ctr;
                    fv_wr_data <= use_relu ? act_out :
                                  act_out; // FC3: pass raw (mac_unit handles)
                    fv_wr_en   <= 1;
                    state      <= S5;
                end

                // State 5: check if all neurons done
                S5: begin
                    if (neuron_ctr == n_out_cur - 1) begin
                        neuron_ctr <= 0;
                        // Move to next FC layer
                        if (state == S5) begin
                            // FC1 done → FC2
                            n_in_cur    <= N_FC2_IN;
                            n_out_cur   <= N_FC2_OUT;
                            wt_base     <= N_FC1_IN * N_FC1_OUT;
                            bias_base   <= N_FC1_OUT;
                            fv_in_base  <= 16'd256;
                            fv_out_base <= 16'd256 + N_FC1_OUT;
                            use_relu    <= 1;
                            state       <= S6;
                        end
                    end else begin
                        neuron_ctr <= neuron_ctr + 1;
                        state      <= S2;
                    end
                end

                // ── FC2 States (States 6-9) ──────────────────
                S6: begin
                    bias_rd_addr <= bias_base + neuron_ctr[9:0];
                    state        <= S7;
                end

                S7: begin
                    if (input_ctr == 0) bias_en <= 1;
                    fv_rd_addr <= fv_in_base + input_ctr;
                    wt_rd_addr <= wt_base + neuron_ctr * n_in_cur + input_ctr;
                    load_acc   <= 1;
                    input_ctr  <= input_ctr + 1;
                    if (input_ctr == n_in_cur - 1) begin
                        input_ctr <= 0;
                        state     <= S8;
                    end
                end

                S8: begin
                    fv_wr_addr <= fv_out_base + neuron_ctr;
                    fv_wr_data <= act_out;
                    fv_wr_en   <= 1;
                    state      <= S9;
                end

                S9: begin
                    if (neuron_ctr == n_out_cur - 1) begin
                        neuron_ctr  <= 0;
                        // FC2 done → FC3
                        n_in_cur    <= N_FC3_IN;
                        n_out_cur   <= N_FC3_OUT;
                        wt_base     <= N_FC1_IN*N_FC1_OUT + N_FC2_IN*N_FC2_OUT;
                        bias_base   <= N_FC1_OUT + N_FC2_OUT;
                        fv_in_base  <= 16'd256 + N_FC1_OUT;
                        fv_out_base <= 16'd256 + N_FC1_OUT + N_FC2_OUT;
                        use_relu    <= 0;   // no ReLU at output layer
                        state       <= S10;
                    end else begin
                        neuron_ctr <= neuron_ctr + 1;
                        state      <= S6;
                    end
                end

                // ── FC3 States (States 10-13) ─────────────────
                S10: begin
                    bias_rd_addr <= bias_base + neuron_ctr[9:0];
                    state        <= S11;
                end

                S11: begin
                    if (input_ctr == 0) bias_en <= 1;
                    fv_rd_addr <= fv_in_base + input_ctr;
                    wt_rd_addr <= wt_base + neuron_ctr * n_in_cur + input_ctr;
                    load_acc   <= 1;
                    input_ctr  <= input_ctr + 1;
                    if (input_ctr == n_in_cur - 1) begin
                        input_ctr <= 0;
                        state     <= S12;
                    end
                end

                S12: begin
                    // FC3 output: write raw value (comparator, not ReLU)
                    out_wr_addr <= neuron_ctr[9:0];
                    out_wr_data <= act_out;
                    out_wr_en   <= 1;
                    state       <= S13;
                end

                // State 14 in paper = STOP
                S13: begin
                    if (neuron_ctr == n_out_cur - 1) begin
                        neuron_ctr <= 0;
                        state      <= S14;
                    end else begin
                        neuron_ctr <= neuron_ctr + 1;
                        state      <= S10;
                    end
                end

                S14: begin
                    fc_done <= 1;
                    state   <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
