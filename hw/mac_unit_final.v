


module mac_unit #(
    parameter DATA_W    = 8,
    parameter ACC_W     = 24,
    parameter FRAC_BITS = 6
)(
    input  wire                       clk,
    input  wire                       rst,
    input  wire signed [DATA_W-1:0]   FV,
    input  wire signed [DATA_W-1:0]   W,
    input  wire signed [DATA_W-1:0]   bias,
    input  wire                       bias_en,
    input  wire                       load_acc,
    output wire signed [ACC_W-1:0]    mac_out,
    output wire signed [DATA_W-1:0]   act_out
);
    // Guard against unknown operands propagating into the accumulator.
    wire signed [DATA_W-1:0] fv_safe   = ((^FV)   === 1'bx) ? {DATA_W{1'b0}} : FV;
    wire signed [DATA_W-1:0] w_safe    = ((^W)    === 1'bx) ? {DATA_W{1'b0}} : W;
    wire signed [DATA_W-1:0] bias_safe = ((^bias) === 1'bx) ? {DATA_W{1'b0}} : bias;

    // Pipeline stage 1: multiply
    reg signed [2*DATA_W-1:0] m_out;
    reg signed [ACC_W-1:0] acc_reg;

    wire signed [ACC_W-1:0] m_ext =
        {{(ACC_W-2*DATA_W){m_out[2*DATA_W-1]}}, m_out};
    wire signed [ACC_W-1:0] bias_ext =
        {{(ACC_W-DATA_W){bias_safe[DATA_W-1]}}, bias_safe} <<< FRAC_BITS;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            m_out   <= 0;
            acc_reg <= 0;
        end else if (bias_en) begin
            acc_reg <= bias_ext;
            m_out   <= 0;
        end else if (load_acc) begin
            acc_reg <= acc_reg + m_ext;
            m_out   <= fv_safe * w_safe;
        end
    end

    assign mac_out = acc_reg;

    // ReLU (paper Fig. 5b)
    wire sign_b   = acc_reg[ACC_W-1];
    wire overflow = |acc_reg[ACC_W-2:DATA_W+FRAC_BITS-1];
    assign act_out = sign_b   ? {DATA_W{1'b0}} :
                     overflow ? {1'b0,{(DATA_W-1){1'b1}}} :
                                acc_reg[DATA_W+FRAC_BITS-2:FRAC_BITS];
endmodule
