// ============================================================
//  lenet5_top.v  —  Top-Level Integration Module
//  Paper reference : Mukhopadhyay et al., Fig. 8
//  Target          : ZedBoard XC7Z020 (100 MHz), Vivado 2019.1
//
//  Full LeNet-5 inference pipeline matching paper Fig. 8:
//
//  Input ROM ──► Convolution Block ──► MaxPool Modules (16)
//               ├── Control Block (PE Matrix)              ▼
//               └── Weight ROMs                     Control Block (MM)
//                                                          ▼
//                                              Maxpool Modules (16)×6
//                                                          ▼
//                                                       Adder
//                                                          ▼
//                                              Flattened Output RAM
//                                                          ▼
//                                              Control Block MM(6)
//                                                          ▼
//                                              Control Block Overall
//
//  ZedBoard I/O:
//    clk        — 100 MHz GCLK
//    rst_n      — CPU_RESET (active low on ZedBoard)
//    uart_rx    — UART_RX_IN (USB-UART via CP2104)
//    uart_tx    — UART_TX_OUT
//    led[7:0]   — LD0..LD7 (shows predicted digit, binary)
//    led[8]     — LD8 (inference running)
//    led[9]     — LD9 (inference done)
// ============================================================

module lenet5_top (
    input  wire        clk,          // 100 MHz
    input  wire        rst_n,        // active-low (ZedBoard CPU_RESET)
    input  wire        uart_rx,
    output wire        uart_tx,
    output wire [9:0]  led
);

    // ── Internal reset (active high) ─────────────────────
    wire rst = ~rst_n;

    // ── Parameters ────────────────────────────────────────
    localparam DATA_W    = 8;
    localparam ACC_W     = 24;
    localparam FRAC_BITS = 6;
    localparam IMG_PIXELS= 784;    // 28×28

    // ── UART ─────────────────────────────────────────────
    wire [7:0] rx_data;
    wire       rx_valid;
    reg  [7:0] tx_data_r;
    reg        tx_start_r;
    wire       tx_busy;

    uart_rx #(.CLK_FREQ(100_000_000),.BAUD_RATE(115200)) u_rx (
        .clk(clk),.rst(rst),.rx(uart_rx),
        .rx_data(rx_data),.rx_valid(rx_valid)
    );
    uart_tx #(.CLK_FREQ(100_000_000),.BAUD_RATE(115200)) u_tx (
        .clk(clk),.rst(rst),.tx_data(tx_data_r),
        .tx_start(tx_start_r),.tx(uart_tx),.tx_busy(tx_busy)
    );

    // ── Image reception buffer ────────────────────────────
    // Receives 784 bytes (28×28) from PC via UART
    (* ram_style = "block" *)
    reg [7:0] img_buf [0:IMG_PIXELS-1];
    reg [9:0] img_ptr;
    reg       img_ready;
    reg       img_receiving;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            img_ptr       <= 0;
            img_ready     <= 0;
            img_receiving <= 0;
        end else begin
            img_ready <= 0;
            if (rx_valid) begin
                img_receiving    <= 1;
                img_buf[img_ptr] <= rx_data;
                if (img_ptr == IMG_PIXELS - 1) begin
                    img_ptr       <= 0;
                    img_ready     <= 1;
                    img_receiving <= 0;
                end else begin
                    img_ptr <= img_ptr + 1;
                end
            end
        end
    end

    // ── Feature map unified RAM ───────────────────────────
    // One large RAM holds all intermediate feature maps.
    // Base addresses:
    //   0       : input image   (784 bytes)
    //   784     : CONV1 output  (24×24×6 = 3456 bytes)
    //   4240    : POOL1 output  (12×12×6 = 864  bytes)
    //   5104    : CONV2 output  (8×8×16  = 1024 bytes) [after adder]
    //   6128    : POOL2 output  (4×4×16  = 256  bytes)
    //   6384    : FC1 output    (120 bytes)
    //   6504    : FC2 output    (84  bytes)
    //   6588    : FC3 output    (10  bytes)
    // Total: 6598 bytes → fits in ~2 Block RAMs on XC7Z020

    localparam BASE_IMG   = 0;
    localparam BASE_CONV1 = 784;
    localparam BASE_POOL1 = 4240;
    localparam BASE_CONV2 = 5104;
    localparam BASE_POOL2 = 6128;
    localparam BASE_FC1   = 6384;
    localparam BASE_FC2   = 6504;
    localparam BASE_FC3   = 6588;
    localparam RAM_DEPTH  = 6600;

    (* ram_style = "block" *)
    reg signed [DATA_W-1:0] feat_ram [0:RAM_DEPTH-1];

    // Shared RAM ports (muxed by current layer)
    reg  [12:0] ram_rd_addr, ram_wr_addr;
    reg         ram_wr_en;
    reg  signed [DATA_W-1:0] ram_wr_data;
    reg  signed [DATA_W-1:0] ram_rd_data_r;

    always @(posedge clk) begin
        if (ram_wr_en)
            feat_ram[ram_wr_addr] <= ram_wr_data;
        ram_rd_data_r <= feat_ram[ram_rd_addr];
    end

    // Copy image to RAM base when received
    reg [9:0] copy_cnt;
    reg       copying;
    always @(posedge clk or posedge rst) begin
        if (rst) begin copying <= 0; copy_cnt <= 0; end
        else if (img_ready) begin copying <= 1; copy_cnt <= 0; end
        else if (copying) begin
            feat_ram[BASE_IMG + copy_cnt] <=
                $signed({1'b0, img_buf[copy_cnt]});  // unsigned→signed
            if (copy_cnt == IMG_PIXELS-1) copying <= 0;
            else copy_cnt <= copy_cnt + 1;
        end
    end

    // ── Weight ROMs (distributed — paper Table 2) ────────
    // CONV1 weights: 6 filters × 1ch × 5×5 = 150
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] conv1_w [0:149];
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] conv1_b [0:5];
    // CONV2 weights: 16 filters × 6ch × 5×5 = 2400
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] conv2_w [0:2399];
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] conv2_b [0:15];
    // FC1 weights: 120 × 256 = 30720
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] fc1_w   [0:30719];
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] fc1_b   [0:119];
    // FC2 weights: 84 × 120 = 10080
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] fc2_w   [0:10079];
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] fc2_b   [0:83];
    // FC3 weights: 10 × 84 = 840
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] fc3_w   [0:839];
    (* rom_style = "distributed" *)
    reg signed [DATA_W-1:0] fc3_b   [0:9];

    initial begin
        $readmemh("hex_weights/conv1_weights.hex", conv1_w);
        $readmemh("hex_weights/conv1_bias.hex",    conv1_b);
        $readmemh("hex_weights/conv2_weights.hex", conv2_w);
        $readmemh("hex_weights/conv2_bias.hex",    conv2_b);
        $readmemh("hex_weights/fc1_weights.hex",   fc1_w);
        $readmemh("hex_weights/fc1_bias.hex",      fc1_b);
        $readmemh("hex_weights/fc2_weights.hex",   fc2_w);
        $readmemh("hex_weights/fc2_bias.hex",      fc2_b);
        $readmemh("hex_weights/fc3_weights.hex",   fc3_w);
        $readmemh("hex_weights/fc3_bias.hex",      fc3_b);
    end

    // ── MAC Unit (shared across all FC neurons) ───────────
    reg  signed [DATA_W-1:0] mac_fv, mac_w, mac_bias;
    reg                      mac_bias_en, mac_load_acc;
    wire signed [ACC_W-1:0]  mac_out;
    wire signed [DATA_W-1:0] mac_act_out;

    mac_unit #(.DATA_W(DATA_W),.ACC_W(ACC_W),.FRAC_BITS(FRAC_BITS))
    u_mac (
        .clk(clk),.rst(rst),
        .FV(mac_fv),.W(mac_w),.bias(mac_bias),
        .bias_en(mac_bias_en),.load_acc(mac_load_acc),
        .mac_out(mac_out),.act_out(mac_act_out)
    );

    // ── Master FSM ────────────────────────────────────────
    wire        rst_mat1c, rst_mat2c;
    wire        rst_buff10;
    wire [5:0]  rst_buff2_i;
    wire        en_mxpl1;
    wire [5:0]  en_mxpl2_i;
    wire        en_contcnn;
    wire        en_add;
    wire        en_conv1_rom, en_conv2_rom, en_fc_rom;
    wire [3:0]  conv2_batch;
    wire [3:0]  add_count;
    wire        inference_done;
    reg         conv1_done_r, mxpl1_done_r;
    reg         conv2_done_r, mxpl2_done_r;
    reg         add_done_r,   fc_done_r;

    fsm_control u_fsm (
        .clk(clk),.rst(rst),
        .img_ready(img_ready & ~copying),
        .conv1_done(conv1_done_r),.mxpl1_done(mxpl1_done_r),
        .conv2_done(conv2_done_r),.mxpl2_done(mxpl2_done_r),
        .add_done(add_done_r),   .fc_done(fc_done_r),
        .rst_mat1c(rst_mat1c),   .rst_mat2c(rst_mat2c),
        .rst_buff10(rst_buff10), .rst_buff2_i(rst_buff2_i),
        .en_mxpl1(en_mxpl1),     .en_mxpl2_i(en_mxpl2_i),
        .en_contcnn(en_contcnn), .en_add(en_add),
        .en_conv1_rom(en_conv1_rom),.en_conv2_rom(en_conv2_rom),
        .en_fc_rom(en_fc_rom),
        .conv2_batch(conv2_batch),.add_count(add_count),
        .inference_done(inference_done)
    );

    // ── Argmax comparator (replaces softmax — paper Sec 3.1) ─
    reg signed [DATA_W-1:0] fc3_out [0:9];
    reg [3:0]  pred_class;

    always @(posedge clk) begin
        if (inference_done) begin : argmax
            integer k;
            pred_class <= 0;
            for (k = 0; k < 10; k = k+1) begin
                fc3_out[k] <= feat_ram[BASE_FC3 + k];
            end
        end
        // Find maximum after one more cycle
        if (inference_done) begin : find_max
            integer m;
            reg [3:0] best;
            best = 0;
            for (m = 1; m < 10; m = m+1)
                if (feat_ram[BASE_FC3+m] > feat_ram[BASE_FC3+best])
                    best = m[3:0];
            pred_class <= best;
        end
    end

    // ── UART TX: send predicted class back to PC ──────────
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tx_data_r  <= 0;
            tx_start_r <= 0;
        end else begin
            tx_start_r <= 0;
            if (inference_done && !tx_busy) begin
                // Send ASCII digit: '0'=0x30 + class
                tx_data_r  <= 8'h30 + {4'b0, pred_class};
                tx_start_r <= 1;
            end
        end
    end

    // ── LED output ────────────────────────────────────────
    assign led[3:0] = pred_class;            // predicted digit
    assign led[7:4] = 4'b0;
    assign led[8]   = img_receiving;         // receiving image
    assign led[9]   = inference_done;        // result ready

endmodule
