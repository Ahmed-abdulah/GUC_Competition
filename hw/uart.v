// ============================================================
//  uart_rx.v  —  UART Receiver
//  Target : ZedBoard XC7Z020 (100 MHz), Vivado 2019.1
//  Config : 115200 baud, 8-N-1
// ============================================================
module uart_rx #(
    parameter CLK_FREQ  = 100_000_000,
    parameter BAUD_RATE = 115_200
)(
    input  wire       clk, rst,
    input  wire       rx,
    output reg  [7:0] rx_data,
    output reg        rx_valid
);
    localparam CLK_PER_BIT = CLK_FREQ / BAUD_RATE;
    localparam IDLE=2'd0, START=2'd1, DATA=2'd2, STOP=2'd3;

    reg [1:0]  state;
    reg [15:0] cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shift;
    reg rx1, rx2;

    always @(posedge clk) begin rx1 <= rx; rx2 <= rx1; end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=IDLE; cnt<=0; bit_idx<=0;
            shift<=0; rx_data<=0; rx_valid<=0;
        end else begin
            rx_valid <= 0;
            case (state)
                IDLE:  if (!rx2) begin state<=START; cnt<=0; end
                START: if (cnt==CLK_PER_BIT/2-1) begin
                           state <= rx2 ? IDLE : DATA;
                           cnt<=0; bit_idx<=0;
                       end else cnt<=cnt+1;
                DATA:  if (cnt==CLK_PER_BIT-1) begin
                           cnt<=0;
                           shift[bit_idx]<=rx2;
                           if (bit_idx==7) state<=STOP;
                           else bit_idx<=bit_idx+1;
                       end else cnt<=cnt+1;
                STOP:  if (cnt==CLK_PER_BIT-1) begin
                           rx_valid<=1; rx_data<=shift;
                           state<=IDLE; cnt<=0;
                       end else cnt<=cnt+1;
            endcase
        end
    end
endmodule

// ============================================================
//  uart_tx.v  —  UART Transmitter
// ============================================================
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
    localparam CLK_PER_BIT = CLK_FREQ / BAUD_RATE;
    localparam IDLE=2'd0, START=2'd1, DATA=2'd2, STOP=2'd3;

    reg [1:0]  state;
    reg [15:0] cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shift;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=IDLE; tx<=1; tx_busy<=0; cnt<=0; bit_idx<=0; shift<=0;
        end else begin
            case (state)
                IDLE:  begin tx<=1; tx_busy<=0;
                           if (tx_start) begin shift<=tx_data; tx_busy<=1; state<=START; cnt<=0; end
                       end
                START: begin tx<=0;
                           if (cnt==CLK_PER_BIT-1) begin cnt<=0; bit_idx<=0; state<=DATA; end
                           else cnt<=cnt+1;
                       end
                DATA:  begin tx<=shift[bit_idx];
                           if (cnt==CLK_PER_BIT-1) begin
                               cnt<=0;
                               if (bit_idx==7) state<=STOP; else bit_idx<=bit_idx+1;
                           end else cnt<=cnt+1;
                       end
                STOP:  begin tx<=1;
                           if (cnt==CLK_PER_BIT-1) begin state<=IDLE; cnt<=0; end
                           else cnt<=cnt+1;
                       end
            endcase
        end
    end
endmodule
