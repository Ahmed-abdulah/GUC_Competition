

module uart_rx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115_200
)(
    input  wire       clk, rst,
    input  wire       rx,
    output reg  [7:0] rx_data,
    output reg        rx_valid
);
    localparam CPB = CLK_FREQ/BAUD_RATE;
    localparam IDLE=2'd0,START=2'd1,DATA=2'd2,STOP=2'd3;
    reg [1:0]  st; reg [15:0] cnt; reg [2:0] bi; reg [7:0] sr;
    reg rx1,rx2;
    always @(posedge clk) begin rx1<=rx; rx2<=rx1; end
    always @(posedge clk or posedge rst) begin
        if(rst) begin st<=IDLE;cnt<=0;bi<=0;sr<=0;rx_data<=0;rx_valid<=0; end
        else begin
            rx_valid<=0;
            case(st)
                IDLE:  if(!rx2) begin st<=START;cnt<=0; end
                START: if(cnt==CPB/2-1) begin st<=rx2?IDLE:DATA;cnt<=0;bi<=0; end
                       else cnt<=cnt+1;
                DATA:  if(cnt==CPB-1) begin cnt<=0;sr[bi]<=rx2;
                           if(bi==7) st<=STOP; else bi<=bi+1;
                       end else cnt<=cnt+1;
                STOP:  if(cnt==CPB-1) begin rx_valid<=1;rx_data<=sr;st<=IDLE;cnt<=0; end
                       else cnt<=cnt+1;
            endcase
        end
    end
endmodule

module uart_tx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115_200
)(
    input  wire       clk, rst,
    input  wire [7:0] tx_data,
    input  wire       tx_start,
    output reg        tx,
    output reg        tx_busy
);
    localparam CPB = CLK_FREQ/BAUD_RATE;
    localparam IDLE=2'd0,START=2'd1,DATA=2'd2,STOP=2'd3;
    reg [1:0]  st; reg [15:0] cnt; reg [2:0] bi; reg [7:0] sr;
    always @(posedge clk or posedge rst) begin
        if(rst) begin st<=IDLE;tx<=1;tx_busy<=0;cnt<=0;bi<=0;sr<=0; end
        else case(st)
            IDLE:  begin tx<=1;tx_busy<=0;
                       if(tx_start) begin sr<=tx_data;tx_busy<=1;st<=START;cnt<=0; end end
            START: begin tx<=0;
                       if(cnt==CPB-1) begin cnt<=0;bi<=0;st<=DATA; end else cnt<=cnt+1; end
            DATA:  begin tx<=sr[bi];
                       if(cnt==CPB-1) begin cnt<=0;
                           if(bi==7) st<=STOP; else bi<=bi+1;
                       end else cnt<=cnt+1; end
            STOP:  begin tx<=1;
                       if(cnt==CPB-1) begin st<=IDLE;cnt<=0; end else cnt<=cnt+1; end
        endcase
    end
endmodule
