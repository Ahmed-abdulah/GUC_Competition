
module pe_block #(
    parameter DATA_W = 8,
    parameter ACC_W  = 24
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire signed [DATA_W-1:0] In,
    input  wire signed [DATA_W-1:0] W,
    input  wire signed [ACC_W-1:0]  Prev,
    input  wire                     Clr,
    input  wire                     Selp,
    input  wire                     Seln,
    output wire signed [ACC_W-1:0]  Out
);
    // Multiply stage
    wire signed [2*DATA_W-1:0] mult = In * W;
    wire signed [ACC_W-1:0]    mult_ext =
        {{(ACC_W-2*DATA_W){mult[2*DATA_W-1]}}, mult};

    // Selp MUX
    wire signed [ACC_W-1:0] prev_mux = Selp ? Prev : {ACC_W{1'b0}};

    // Adder + register
    wire signed [ACC_W-1:0] sum = mult_ext + prev_mux;
    reg  signed [ACC_W-1:0] reg_out;

    always @(posedge clk) begin
        if (rst || Clr)
            reg_out <= {ACC_W{1'b0}};
        else
            reg_out <= sum;
    end

    // Seln MUX — pass or hold zero
    assign Out = Seln ? reg_out : {ACC_W{1'b0}};
endmodule
