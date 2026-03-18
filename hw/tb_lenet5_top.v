// ============================================================
//  tb_lenet5_top.v  —  Top-Level Testbench
//  Target : ZedBoard XC7Z020, Vivado 2019.1
//
//  Sends one MNIST test image over simulated UART,
//  waits for predicted class response.
// ============================================================
`timescale 1ns/1ps

module tb_lenet5_top;

    parameter CLK_PERIOD = 10;         // 100 MHz → 10 ns
    parameter BAUD_PERIOD = 8681;      // 115200 baud → ~8681 ns

    reg        clk, rst_n;
    reg        uart_rx_tb;
    wire       uart_tx_tb;
    wire [9:0] led;

    // ── DUT ──────────────────────────────────────────────
    lenet5_top dut (
        .clk(clk), .rst_n(rst_n),
        .uart_rx(uart_rx_tb),
        .uart_tx(uart_tx_tb),
        .led(led)
    );

    // ── Clock ─────────────────────────────────────────────
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ── UART send task ────────────────────────────────────
    task uart_send_byte;
        input [7:0] data;
        integer i;
        begin
            uart_rx_tb = 0;              // start bit
            #BAUD_PERIOD;
            for (i = 0; i < 8; i = i+1) begin
                uart_rx_tb = data[i];    // LSB first
                #BAUD_PERIOD;
            end
            uart_rx_tb = 1;              // stop bit
            #BAUD_PERIOD;
        end
    endtask

    // ── Test image: a simple pattern for '0' ──────────────
    reg [7:0] test_img [0:783];
    integer   idx;

    initial begin
        // Fill with a simple handwritten '0' pattern
        // (all zeros except center ring — minimal test)
        for (idx = 0; idx < 784; idx = idx+1)
            test_img[idx] = 8'h00;
        // Draw a rough circle (digit '0' approximation)
        // Row 8-20, Cols 8-20 outline
        for (idx = 0; idx < 784; idx = idx+1) begin
            if (idx/28 >= 8 && idx/28 <= 20 &&
                idx%28 >= 8 && idx%28 <= 20 &&
                (idx/28 == 8 || idx/28 == 20 ||
                 idx%28 == 8 || idx%28 == 20))
                test_img[idx] = 8'hFF;
        end
    end

    // ── Main test ─────────────────────────────────────────
    initial begin
        $dumpfile("tb_lenet5_top.vcd");
        $dumpvars(0, tb_lenet5_top);

        rst_n      = 0;
        uart_rx_tb = 1;    // UART idle high
        #200;
        rst_n = 1;
        #100;

        $display("[%0t] Sending 784-byte MNIST image via UART...", $time);

        // Send all 784 image bytes
        for (idx = 0; idx < 784; idx = idx+1) begin
            uart_send_byte(test_img[idx]);
        end

        $display("[%0t] Image sent. Waiting for inference...", $time);

        // Wait for inference done signal
        wait(led[9] == 1);
        #100;

        $display("[%0t] Inference complete! Predicted class = %0d",
                  $time, led[3:0]);
        $display("[%0t] LED value = %b", $time, led);

        // Wait for UART response
        #(BAUD_PERIOD * 12);
        $display("[%0t] Simulation complete.", $time);
        $finish;
    end

    // ── Timeout watchdog ─────────────────────────────────
    initial begin
        #50_000_000_000;   // 50 ms timeout
        $display("TIMEOUT: Inference did not complete in time.");
        $finish;
    end

endmodule
