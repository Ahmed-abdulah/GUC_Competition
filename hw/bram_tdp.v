

module bram_tdp #(
    parameter DATA_W = 8,
    parameter DEPTH  = 8192,
    parameter ADDR_W = 13
)(
    input  wire              clk,
    input  wire              ena,
    input  wire              wea,
    input  wire [ADDR_W-1:0] addra,
    input  wire [DATA_W-1:0] dina,
    output reg  [DATA_W-1:0] douta,
    input  wire              enb,
    input  wire              web,
    input  wire [ADDR_W-1:0] addrb,
    input  wire [DATA_W-1:0] dinb,
    output reg  [DATA_W-1:0] doutb
);
    (* ram_style = "block" *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (ena) begin
            if (wea) mem[addra] <= dina;
            douta <= mem[addra];
        end
    end
    always @(posedge clk) begin
        if (enb) begin
            if (web) mem[addrb] <= dinb;
            doutb <= mem[addrb];
        end
    end
endmodule
