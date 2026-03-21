

module conv_maxpool_block #(
    parameter DATA_W    = 8,
    parameter ACC_W     = 24,
    parameter FILT      = 5,
    parameter FRAC_BITS = 6
)(
    input  wire                    clk, rst,
    // Runtime configuration
    input  wire [4:0]              n_filt,
    input  wire [4:0]              in_h,
    input  wire [4:0]              in_w,
    input  wire [2:0]              in_ch,
    // FSM control
    input  wire                    rst_matc,
    input  wire                    rst_buff,
    input  wire                    en_mxpl,
    input  wire                    start,
    // BRAM read (Port B)
    output reg  [15:0]             feat_rd_addr,
    input  wire [DATA_W-1:0]       feat_rd_data,
    // Weight ROM
    output reg  [11:0]             wt_rd_addr,
    input  wire [DATA_W-1:0]       wt_rd_data,
    // Bias ROM
    output reg  [3:0]              bias_addr,
    input  wire [DATA_W-1:0]       bias_data,
    // CONV output to BRAM
    output reg  [11:0]             feat_wr_addr,
    output reg  [DATA_W-1:0]       feat_wr_data,
    output reg                     feat_wr_en,
    // MaxPool output to BRAM
    output reg  [11:0]             mxpl_wr_addr,
    output reg  [DATA_W-1:0]       mxpl_wr_data,
    output reg                     mxpl_wr_en,
    output reg                     pool_rd_active,
    // Status
    output reg                     conv_done,
    output reg                     mxpl_done
);
    wire [4:0] out_h  = in_h - FILT + 1;
    wire [4:0] out_w  = in_w - FILT + 1;
    wire [4:0] pool_h = out_h >> 1;
    wire [4:0] pool_w = out_w >> 1;
    wire [8:0] wt_per = FILT * FILT * {6'b0, in_ch};

    localparam S_IDLE      = 4'd0;
    localparam S_BIAS      = 4'd1;
    localparam S_BIAS_WAIT = 4'd2;
    localparam S_LOAD      = 4'd3;
    localparam S_WAIT      = 4'd4;
    localparam S_MAC       = 4'd5;
    localparam S_RELU      = 4'd6;
    localparam S_NEXT_PIX  = 4'd7;
    localparam S_NEXT_FILT = 4'd8;
    localparam S_POOL_INIT = 4'd9;
    localparam S_POOL_R0   = 4'd10;
    localparam S_POOL_R1   = 4'd11;
    localparam S_POOL_R2   = 4'd12;
    localparam S_POOL_R3   = 4'd13;
    localparam S_POOL_WR   = 4'd14;
    localparam S_POOL_NEXT = 4'd15;

    reg [3:0]  state;
    reg [4:0]  filter_cnt;
    reg [4:0]  out_row, out_col;
    reg [8:0]  mac_idx;
    reg [4:0]  pool_row, pool_col;
    reg signed [ACC_W-1:0] acc;
    reg signed [DATA_W-1:0] px0, px1, px2;

    function [DATA_W-1:0] relu_clip;
        input signed [ACC_W-1:0] x;
        begin
            if (x[ACC_W-1])
                relu_clip = {DATA_W{1'b0}};
            else if (|x[ACC_W-2:DATA_W+FRAC_BITS-1])
                relu_clip = {1'b0, {(DATA_W-1){1'b1}}};
            else
                relu_clip = x[DATA_W+FRAC_BITS-2:FRAC_BITS];
        end
    endfunction

    function signed [DATA_W-1:0] max2;
        input signed [DATA_W-1:0] a, b;
        begin max2 = (a > b) ? a : b; end
    endfunction

    wire [17:0] wt_addr_calc  = ({9'b0,filter_cnt} * {9'b0,wt_per}) + {9'b0,mac_idx};
    wire [17:0] fwr_addr_calc = ({7'b0,filter_cnt} * {7'b0,out_h} * {7'b0,out_w})
                               + ({7'b0,out_row} * {7'b0,out_w})
                               + {7'b0,out_col};
    wire [17:0] pool_base     = {13'b0,filter_cnt} * {13'b0,out_h} * {13'b0,out_w};
    wire [11:0] pool_tl_addr  = pool_base[11:0] + ({6'b0,pool_row}*2)*{6'b0,out_w} + ({6'b0,pool_col}*2);
    wire [11:0] pool_tr_addr  = pool_base[11:0] + ({6'b0,pool_row}*2)*{6'b0,out_w} + ({6'b0,pool_col}*2+1);
    wire [11:0] pool_bl_addr  = pool_base[11:0] + ({6'b0,pool_row}*2+1)*{6'b0,out_w} + ({6'b0,pool_col}*2);
    wire [11:0] pool_br_addr  = pool_base[11:0] + ({6'b0,pool_row}*2+1)*{6'b0,out_w} + ({6'b0,pool_col}*2+1);
    wire [11:0] mxpl_out_addr = {3'b0,filter_cnt}*{3'b0,pool_h}*{3'b0,pool_w}
                               + {3'b0,pool_row}*{3'b0,pool_w}
                               + {3'b0,pool_col};

    // Combinatorial tap address
    reg [15:0] feat_addr_reg;
    reg [8:0]  tap_pos_c;
    reg [3:0]  tap_row_c, tap_col_c;
    reg [2:0]  tap_ch_c;

    always @(*) begin
        case (in_ch)
            3'd1: begin tap_ch_c=0;              tap_pos_c=mac_idx; end
            3'd6: begin tap_ch_c=mac_idx%6; tap_pos_c=mac_idx/6; end
            default: begin tap_ch_c=0;            tap_pos_c=mac_idx; end
        endcase
        tap_row_c = tap_pos_c / 5;
        tap_col_c = tap_pos_c % 5;
        feat_addr_reg =
            {13'b0,tap_ch_c} * {11'b0,in_h} * {11'b0,in_w}
          + ({11'b0,out_row} + {11'b0,tap_row_c}) * {11'b0,in_w}
          + ({11'b0,out_col} + {11'b0,tap_col_c});
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=S_IDLE; filter_cnt<=0; out_row<=0; out_col<=0;
            mac_idx<=0; pool_row<=0; pool_col<=0; acc<=0;
            px0<=0; px1<=0; px2<=0;
            feat_rd_addr<=0; wt_rd_addr<=0; bias_addr<=0;
            feat_wr_addr<=0; feat_wr_data<=0; feat_wr_en<=0;
            mxpl_wr_addr<=0; mxpl_wr_data<=0; mxpl_wr_en<=0;
            pool_rd_active<=0; conv_done<=0; mxpl_done<=0;
        end else begin
            feat_wr_en<=0; mxpl_wr_en<=0;
            conv_done<=0;  mxpl_done<=0;
            pool_rd_active<=0;  // default off; set in pool read states below

            case(state)
                S_IDLE: begin
                    if (start && !rst_matc) begin
                        filter_cnt<=0; out_row<=0; out_col<=0;
                        state<=S_BIAS;
                    end
                end

                S_BIAS: begin
                    bias_addr <= filter_cnt[3:0];
                    state     <= S_BIAS_WAIT;
                end

                S_BIAS_WAIT: begin
                    acc     <= {{(ACC_W-DATA_W){bias_data[DATA_W-1]}},
                                 bias_data} <<< FRAC_BITS;
                    mac_idx <= 0;
                    state   <= S_LOAD;
                end

                S_LOAD: begin
                    wt_rd_addr   <= wt_addr_calc[11:0];
                    feat_rd_addr <= feat_addr_reg;
                    state        <= S_WAIT;
                end

                S_WAIT: state <= S_MAC;

                S_MAC: begin
                    acc <= acc + $signed(wt_rd_data) * $signed(feat_rd_data);
                    if (mac_idx == wt_per - 1) begin
                        state <= S_RELU;
                    end else begin
                        mac_idx <= mac_idx + 1;
                        state   <= S_LOAD;
                    end
                end

                S_RELU: begin
                    feat_wr_data <= relu_clip(acc);
                    feat_wr_addr <= fwr_addr_calc[11:0];
                    feat_wr_en   <= !rst_buff;
                    state        <= S_NEXT_PIX;
                end

                S_NEXT_PIX: begin
                    if (out_col == out_w - 1) begin
                        out_col <= 0;
                        if (out_row == out_h - 1) begin
                            out_row <= 0;
                            state   <= S_NEXT_FILT;
                        end else begin
                            out_row <= out_row + 1;
                            state   <= S_BIAS;
                        end
                    end else begin
                        out_col <= out_col + 1;
                        state   <= S_BIAS;
                    end
                end

                S_NEXT_FILT: begin
                    if (filter_cnt == n_filt - 1) begin
                        conv_done  <= 1;
                        filter_cnt <= 0;
                        if (en_mxpl) begin
                            pool_row <= 0; pool_col <= 0;
                            state    <= S_POOL_INIT;
                        end else
                            state <= S_IDLE;
                    end else begin
                        filter_cnt <= filter_cnt + 1;
                        out_row    <= 0; out_col <= 0;
                        state      <= S_BIAS;
                    end
                end

                S_POOL_INIT: begin
                    pool_rd_active <= 1;
                    feat_rd_addr   <= {4'b0, pool_tl_addr};
                    state          <= S_POOL_R0;
                end

                S_POOL_R0: begin
                    pool_rd_active <= 1;
                    feat_rd_addr   <= {4'b0, pool_tr_addr};
                    state          <= S_POOL_R1;
                end

                S_POOL_R1: begin
                    pool_rd_active <= 1;
                    px0            <= feat_rd_data;
                    feat_rd_addr   <= {4'b0, pool_bl_addr};
                    state          <= S_POOL_R2;
                end

                S_POOL_R2: begin
                    pool_rd_active <= 1;
                    px1            <= feat_rd_data;
                    feat_rd_addr   <= {4'b0, pool_br_addr};
                    state          <= S_POOL_R3;
                end

                S_POOL_R3: begin
                    pool_rd_active <= 1;
                    px2            <= feat_rd_data;
                    state          <= S_POOL_WR;
                end

                S_POOL_WR: begin
                    mxpl_wr_data <= max2(max2(px0,px1), max2(px2,feat_rd_data));
                    mxpl_wr_addr <= mxpl_out_addr;
                    mxpl_wr_en   <= 1;
                    state        <= S_POOL_NEXT;
                end

                S_POOL_NEXT: begin
                    mxpl_wr_en <= 1;
                    if (pool_col == pool_w - 1) begin
                        pool_col <= 0;
                        if (pool_row == pool_h - 1) begin
                            pool_row <= 0;
                            if (filter_cnt == n_filt - 1) begin
                                filter_cnt <= 0;
                                mxpl_done  <= 1;
                                state      <= S_IDLE;
                            end else begin
                                filter_cnt <= filter_cnt + 1;
                                state      <= S_POOL_INIT;
                            end
                        end else begin
                            pool_row <= pool_row + 1;
                            state    <= S_POOL_INIT;
                        end
                    end else begin
                        pool_col <= pool_col + 1;
                        state    <= S_POOL_INIT;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
