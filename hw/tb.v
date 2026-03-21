
`timescale 1ns/1ps

module tb;

    localparam CLK_PERIOD  = 10;
    localparam BAUD_CYCLES = 868;

    // ── DUT ──────────────────────────────────────────────────
    reg        clk, rst_n, uart_rx_tb;
    wire       uart_tx_tb;
    wire [9:0] led;

    lenet5_top dut (
        .clk    (clk),   .rst_n  (rst_n),
        .uart_rx(uart_rx_tb), .uart_tx(uart_tx_tb),
        .led    (led)
    );

    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    reg [7:0] img [0:783];
    integer   i, b;

    // ── Result storage ────────────────────────────────────────
    reg [3:0] r_pred [0:9];
    reg [3:0] r_exp  [0:9];
    reg [7:0] r_fc3  [0:9][0:9];

    // ── UART task ─────────────────────────────────────────────
    task send_byte;
        input [7:0] d;
        begin
            uart_rx_tb = 0;
            repeat(BAUD_CYCLES) @(posedge clk);
            for (b = 0; b < 8; b = b+1) begin
                uart_rx_tb = d[b];
                repeat(BAUD_CYCLES) @(posedge clk);
            end
            uart_rx_tb = 1;
            repeat(BAUD_CYCLES) @(posedge clk);
        end
    endtask

    // ── Reset task ────────────────────────────────────────────
    task do_reset;
        begin
            rst_n = 0;
            repeat(500) @(posedge clk);
            rst_n = 1;
            repeat(100) @(posedge clk);
        end
    endtask

    // ── Run one inference ─────────────────────────────────────
    task run_case;
        input [3:0] case_idx;
        input [3:0] expected;
        integer jj;
        begin
            do_reset;

            $display("");
            $display("============================================================");
            $display("  CASE %0d  |  digit%0d.png  |  expected = %0d",
                case_idx, case_idx, expected);
            $display("============================================================");

            if (case_idx == 0) begin
                $display("  [WEIGHT] conv1_w[0]=0x%02X  fc3_bias[6]=0x%02X",
                    dut.u_conv1_w_rom.mem[0], dut.u_fc3_b_rom.mem[6]);
                if (dut.u_conv1_w_rom.mem[0] === 8'h00)
                    $display("  [ERROR] Weights ZERO — hex_weights/ missing!");
                else
                    $display("  [OK] Weights loaded");
            end

            for (i = 0; i < 784; i = i+1)
                send_byte(img[i]);
            $display("  Sent 784 bytes. pixel[0]=%0d", img[0]);

            fork
                begin : wd
                    @(posedge led[9]);
                    disable to;
                end
                begin : to
                    repeat(50_000_000) @(posedge clk);
                    $display("  [TIMEOUT]");
                    disable wd;
                end
            join

            repeat(300) @(posedge clk);

            $display("  FC3 [6588..6597] = %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                dut.u_feat_ram.mem[6588], dut.u_feat_ram.mem[6589],
                dut.u_feat_ram.mem[6590], dut.u_feat_ram.mem[6591],
                dut.u_feat_ram.mem[6592], dut.u_feat_ram.mem[6593],
                dut.u_feat_ram.mem[6594], dut.u_feat_ram.mem[6595],
                dut.u_feat_ram.mem[6596], dut.u_feat_ram.mem[6597]);

            r_pred[case_idx] = dut.pred_class;
            r_exp [case_idx] = expected;
            for (jj = 0; jj <= 9; jj = jj+1)
                r_fc3[case_idx][jj] = dut.u_feat_ram.mem[6588+jj];

            $display("  RESULT: predicted=%0d  expected=%0d  %s",
                dut.pred_class, expected,
                (dut.pred_class == expected) ? "PASS" : "FAIL");
        end
    endtask

    // ── MAIN ──────────────────────────────────────────────────
    initial begin
        uart_rx_tb = 1; rst_n = 0;

        $readmemh("sim_image0.hex", img); run_case(4'd0, 4'd0);
        $readmemh("sim_image1.hex", img); run_case(4'd1, 4'd1);
        $readmemh("sim_image2.hex", img); run_case(4'd2, 4'd2);
        $readmemh("sim_image3.hex", img); run_case(4'd3, 4'd3);
        $readmemh("sim_image4.hex", img); run_case(4'd4, 4'd4);
        $readmemh("sim_image5.hex", img); run_case(4'd5, 4'd5);
        $readmemh("sim_image6.hex", img); run_case(4'd6, 4'd6);
        $readmemh("sim_image7.hex", img); run_case(4'd7, 4'd7);
        $readmemh("sim_image8.hex", img); run_case(4'd8, 4'd8);
        $readmemh("sim_image9.hex", img); run_case(4'd9, 4'd9);

        // ── Summary ───────────────────────────────────────────
        $display("");
        $display("============================================================");
        $display("  FINAL SUMMARY — 10 DIGITS");
        $display("============================================================");
        $display("  Case | Exp | RTL | FC3[0..9] unsigned                  | Status");
        $display("  ─────┼─────┼─────┼────────────────────────────────────┼──────");

        begin : sumblk
            integer tt;
            integer pass_count;
            pass_count = 0;
            for (tt = 0; tt < 10; tt = tt+1) begin
                $display("   %0d    |  %0d  |  %0d  | %3d %3d %3d %3d %3d %3d %3d %3d %3d %3d | %s",
                    tt, r_exp[tt], r_pred[tt],
                    r_fc3[tt][0], r_fc3[tt][1], r_fc3[tt][2],
                    r_fc3[tt][3], r_fc3[tt][4], r_fc3[tt][5],
                    r_fc3[tt][6], r_fc3[tt][7], r_fc3[tt][8],
                    r_fc3[tt][9],
                    (r_pred[tt] == r_exp[tt]) ? "PASS" : "FAIL");
                if (r_pred[tt] == r_exp[tt])
                    pass_count = pass_count + 1;
            end
            $display("============================================================");
            $display("  Score: %0d / 10  (%0d%%)", pass_count, pass_count*10);
            $display("============================================================");
        end

        #1000; $finish;
    end

    // Watchdog: 10 × 100ms = 1s + margin
    initial begin
        #4_000_000_000;
        $display("[WATCHDOG] 4s limit reached");
        $finish;
    end

endmodule
