

module rom_module #(
    parameter DATA_W    = 8,
    parameter DEPTH     = 256,
    parameter ADDR_W    = 8,
    parameter INIT_FILE = "rom.hex"
)(
    input  wire              clk,
    input  wire              en,
    input  wire [ADDR_W-1:0] addr,
    output reg  [DATA_W-1:0] dout0,
    output reg  [DATA_W-1:0] dout1,
    output reg  [DATA_W-1:0] dout2,
    output reg  [DATA_W-1:0] dout3,
    output reg  [DATA_W-1:0] dout4
);
    (* rom_style = "distributed" *)
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    initial $readmemh(INIT_FILE, mem);

    always @(posedge clk) begin
        if (en) begin
            dout0 <= mem[addr];
            dout1 <= (addr+1 < DEPTH) ? mem[addr+1] : {DATA_W{1'b0}};
            dout2 <= (addr+2 < DEPTH) ? mem[addr+2] : {DATA_W{1'b0}};
            dout3 <= (addr+3 < DEPTH) ? mem[addr+3] : {DATA_W{1'b0}};
            dout4 <= (addr+4 < DEPTH) ? mem[addr+4] : {DATA_W{1'b0}};
        end
    end
endmodule
