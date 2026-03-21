
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
    input  wire                  clk, rst,
    input  wire                  en_contcnn,
    output reg  [12:0]           fv_rd_addr,
    input  wire [DATA_W-1:0]     fv_rd_data,
    input  wire                  fv_rd_en,
    output reg  [17:0]           wt_rd_addr,
    input  wire [DATA_W-1:0]     wt_rd_data,
    output reg  [9:0]            bias_rd_addr,
    input  wire [DATA_W-1:0]     bias_rd_data,
    output reg  [9:0]            out_wr_addr,
    output reg  [DATA_W-1:0]     out_wr_data,
    output reg                   out_wr_en,
    output reg  [12:0]           fv_wr_addr,
    output reg  [DATA_W-1:0]     fv_wr_data,
    output reg                   fv_wr_en,
    output reg                   bias_en,
    output reg                   load_acc,
    input  wire [DATA_W-1:0]     act_out,
    output reg                   fc_done
);
    // State codes (5 bits) — *_D = addr, *_WAIT = BRAM/ROM 1-cycle, *_E = MAC
    localparam S_IDLE=5'd0,  S3WAIT=5'd1, S7WAIT=5'd3, S11WAIT=5'd7,
               S2=5'd2,   S2BWAIT=5'd11, S6BWAIT=5'd30, S10BWAIT=5'd31,
               S2W=5'd21,
               S3D=5'd22, S3E=5'd23,
               S4=5'd4,   S4a=5'd15, S4b=5'd16,
               S5=5'd5,   S6=5'd6,   S6W=5'd24,
               S7D=5'd25, S7E=5'd26,
               S8=5'd8,   S8a=5'd17, S8b=5'd18,
               S9=5'd9,   S10=5'd10, S10W=5'd27,
               S11D=5'd28,S11E=5'd29,
               S12=5'd12, S12a=5'd19,S12b=5'd20,
               S13=5'd13, S14=5'd14;

    reg [4:0]  state;
    reg [15:0] input_ctr;
    reg [9:0]  neuron_ctr;
    reg [15:0] n_in;
    reg [9:0]  n_out;
    reg [17:0] wt_base;
    reg [9:0]  bias_base;
    reg [12:0] fv_in_base, fv_out_base;
    reg        use_relu;

    wire [25:0] wt_offset = {16'b0, neuron_ctr} * {10'b0, n_in};
    wire [17:0] wt_addr_full = wt_base + wt_offset[17:0] + {2'b0, input_ctr[15:0]};

    wire [12:0] neuron_ext = {3'b0, neuron_ctr};

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=S_IDLE; fc_done<=0; bias_en<=0;
            load_acc<=0; out_wr_en<=0; fv_wr_en<=0;
            input_ctr<=0; neuron_ctr<=0;
            fv_rd_addr<=0; wt_rd_addr<=0; bias_rd_addr<=0;
            out_wr_addr<=0; out_wr_data<=0;
            fv_wr_addr<=0; fv_wr_data<=0;
            n_in<=0; n_out<=0; wt_base<=0; bias_base<=0;
            fv_in_base<=0; fv_out_base<=0; use_relu<=0;
        end else begin
            bias_en<=0; load_acc<=0;
            out_wr_en<=0; fv_wr_en<=0; fc_done<=0;

            case(state)
                S_IDLE: if(en_contcnn) begin
                    n_in        <= N_FC1_IN;
                    n_out       <= N_FC1_OUT;
                    wt_base     <= 18'd0;
                    bias_base   <= 10'd0;
                    fv_in_base  <= 13'd6128;
                    fv_out_base <= 13'd6384;
                    use_relu    <= 1;
                    neuron_ctr  <= 0;
                    input_ctr   <= 0;
                    state       <= S2;
                end

                S2: begin
                    bias_rd_addr <= bias_base + neuron_ctr;
                    state        <= S2BWAIT;
                end
                S2BWAIT: state <= S2W;

                S2W: begin
                    bias_en <= 1;
                    state   <= S3D;
                end

                S3D: begin
                    fv_rd_addr <= fv_in_base + input_ctr[12:0];
                    wt_rd_addr <= wt_addr_full;
                    state      <= S3WAIT;
                end
                S3WAIT: state <= S3E;

                S3E: begin
                    load_acc <= 1;
                    if (input_ctr == n_in - 1) begin
                        input_ctr <= 0;
                        state     <= S4;
                    end else begin
                        input_ctr <= input_ctr + 1;
                        state     <= S3D;
                    end
                end

                S4: begin
                    load_acc <= 1;
                    state    <= S4a;
                end
                S4a: begin
                    state <= S4b;
                end
                S4b: begin
                    fv_wr_addr <= fv_out_base + neuron_ext;
                    fv_wr_data <= act_out;
                    fv_wr_en   <= 1;
                    state      <= S5;
                end

                S5: begin
                    if (neuron_ctr == n_out-1) begin
                        neuron_ctr  <= 0;
                        n_in        <= N_FC2_IN;
                        n_out       <= N_FC2_OUT;
                        wt_base     <= N_FC1_IN * N_FC1_OUT;
                        bias_base   <= N_FC1_OUT;
                        fv_in_base  <= 13'd6384;
                        fv_out_base <= 13'd6504;
                        use_relu    <= 1;
                        input_ctr   <= 0;
                        state       <= S6;
                    end else begin
                        neuron_ctr <= neuron_ctr + 1;
                        input_ctr  <= 0;
                        state      <= S2;
                    end
                end

                S6: begin
                    bias_rd_addr <= bias_base + neuron_ctr;
                    state        <= S6BWAIT;
                end
                S6BWAIT: state <= S6W;
                S6W: begin
                    bias_en <= 1;
                    state   <= S7D;
                end
                S7D: begin
                    fv_rd_addr <= fv_in_base + input_ctr[12:0];
                    wt_rd_addr <= wt_addr_full;
                    state      <= S7WAIT;
                end
                S7WAIT: state <= S7E;
                S7E: begin
                    load_acc <= 1;
                    if (input_ctr == n_in - 1) begin
                        input_ctr <= 0;
                        state     <= S8;
                    end else begin
                        input_ctr <= input_ctr + 1;
                        state     <= S7D;
                    end
                end

                S8: begin
                    load_acc <= 1;
                    state    <= S8a;
                end
                S8a: begin
                    state <= S8b;
                end
                S8b: begin
                    fv_wr_addr <= fv_out_base + neuron_ext;
                    fv_wr_data <= act_out;
                    fv_wr_en   <= 1;
                    state      <= S9;
                end

                S9: begin
                    if (neuron_ctr == n_out-1) begin
                        neuron_ctr  <= 0;
                        n_in        <= N_FC3_IN;
                        n_out       <= N_FC3_OUT;
                        wt_base     <= N_FC1_IN*N_FC1_OUT + N_FC2_IN*N_FC2_OUT;
                        bias_base   <= N_FC1_OUT + N_FC2_OUT;
                        fv_in_base  <= 13'd6504;
                        fv_out_base <= 13'd6588;
                        use_relu    <= 0;
                        input_ctr   <= 0;
                        state       <= S10;
                    end else begin
                        neuron_ctr <= neuron_ctr + 1;
                        input_ctr  <= 0;
                        state      <= S6;
                    end
                end

                S10: begin
                    bias_rd_addr <= bias_base + neuron_ctr;
                    state        <= S10BWAIT;
                end
                S10BWAIT: state <= S10W;
                S10W: begin
                    bias_en <= 1;
                    state   <= S11D;
                end
                S11D: begin
                    fv_rd_addr <= fv_in_base + input_ctr[12:0];
                    wt_rd_addr <= wt_addr_full;
                    state      <= S11WAIT;
                end
                S11WAIT: state <= S11E;
                S11E: begin
                    load_acc <= 1;
                    if (input_ctr == n_in - 1) begin
                        input_ctr <= 0;
                        state     <= S12;
                    end else begin
                        input_ctr <= input_ctr + 1;
                        state     <= S11D;
                    end
                end

                S12: begin
                    load_acc <= 1;
                    state    <= S12a;
                end
                S12a: begin
                    state <= S12b;
                end
                S12b: begin
                    out_wr_addr <= neuron_ctr;
                    out_wr_data <= act_out;
                    out_wr_en   <= 1;
                    state       <= S13;
                end

                S13: begin
                    if (neuron_ctr == n_out-1) begin
                        neuron_ctr <= 0;
                        state      <= S14;
                    end else begin
                        neuron_ctr <= neuron_ctr + 1;
                        input_ctr  <= 0;
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
