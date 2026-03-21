

module lenet5_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        uart_rx,
    output wire        uart_tx,
    output wire [9:0]  led
);
    wire rst = ~rst_n;

    localparam DATA_W   = 8;
    localparam ACC_W    = 24;
    localparam FRAC_BITS= 6;
    localparam IMG_PIX  = 784;

    localparam [12:0] BASE_IMG  = 13'd0;
    localparam [12:0] BASE_CONV1= 13'd784;
    localparam [12:0] BASE_POOL1= 13'd4240;
    localparam [12:0] BASE_CONV2= 13'd5104;
    localparam [12:0] BASE_POOL2= 13'd6128;
    localparam [12:0] BASE_FC1  = 13'd6384;
    localparam [12:0] BASE_FC2  = 13'd6504;
    localparam [12:0] BASE_FC3  = 13'd6588;
    localparam        RAM_DEPTH = 8192;

    // =========================================================
    //  UART
    // =========================================================
    wire [7:0] rx_data;
    wire       rx_valid;
    reg  [7:0] tx_data_r;
    reg        tx_start_r;
    wire       tx_busy;

    uart_rx #(.CLK_FREQ(100_000_000),.BAUD_RATE(115_200)) u_rx (
        .clk(clk),.rst(rst),.rx(uart_rx),
        .rx_data(rx_data),.rx_valid(rx_valid)
    );
    uart_tx #(.CLK_FREQ(100_000_000),.BAUD_RATE(115_200)) u_tx (
        .clk(clk),.rst(rst),
        .tx_data(tx_data_r),.tx_start(tx_start_r),
        .tx(uart_tx),.tx_busy(tx_busy)
    );

    // =========================================================
    //  Image buffer + copy FSM
    // =========================================================
    reg [7:0] img_buf [0:IMG_PIX-1];
    reg [9:0] img_ptr;
    reg       img_ready, img_receiving;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            img_ptr<=0; img_ready<=0; img_receiving<=0;
        end else begin
            img_ready <= 0;
            if (rx_valid) begin
                img_receiving    <= 1;
                img_buf[img_ptr] <= rx_data;
                if (img_ptr==IMG_PIX-1) begin
                    img_ptr<=0; img_ready<=1; img_receiving<=0;
                end else img_ptr<=img_ptr+1;
            end
        end
    end

    reg [9:0] copy_cnt;
    reg       copying;
    always @(posedge clk or posedge rst) begin
        if (rst) begin copying<=0; copy_cnt<=0; end
        else if (img_ready) begin copying<=1; copy_cnt<=0; end
        else if (copying) begin
            if (copy_cnt==IMG_PIX-1) copying<=0;
            else copy_cnt<=copy_cnt+1;
        end
    end

    reg prev_copying;
    always @(posedge clk or posedge rst) begin
        if (rst) prev_copying <= 0;
        else      prev_copying <= copying;
    end
    wire img_ready_safe = prev_copying & ~copying;

    // =========================================================
    //  Feature Map BRAM (True Dual-Port, Vivado UG901)
    // =========================================================
    reg  [12:0]      layer_wr_addr;
    reg  [DATA_W-1:0]layer_wr_data;
    reg              layer_wr_en;

    wire [12:0]      bram_a_addr;
    wire [DATA_W-1:0]bram_a_din;
    wire             bram_a_we;
    assign bram_a_addr = copying ? (BASE_IMG+{3'b0,copy_cnt}) : layer_wr_addr;
    assign bram_a_din  = copying ? {1'b0,img_buf[copy_cnt]}   : layer_wr_data;
    assign bram_a_we   = copying | layer_wr_en;

    reg  [12:0]      bram_b_addr;
    wire [DATA_W-1:0]bram_b_dout;

    bram_tdp #(.DATA_W(DATA_W),.DEPTH(RAM_DEPTH),.ADDR_W(13)) u_feat_ram (
        .clk(clk),
        .ena(1'b1),.wea(bram_a_we),.addra(bram_a_addr),
        .dina(bram_a_din),.douta(),
        .enb(1'b1),.web(1'b0),.addrb(bram_b_addr),
        .dinb({DATA_W{1'b0}}),.doutb(bram_b_dout)
    );

    // =========================================================
    //  Distributed Weight/Bias ROMs  (paper Table 2)
    // =========================================================
    // ROM enable wires — declared here so ROM instances below can use them
    wire        en_conv1_rom, en_conv2_rom, en_fc_rom;

    // CONV1 weights (150)
    wire [7:0]       c1w_addr; wire [DATA_W-1:0] c1w_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(150),.ADDR_W(8),
                 .INIT_FILE("hex_weights/conv1_weights.hex")) u_conv1_w_rom (
        .clk(clk),.en(en_conv1_rom),.addr(c1w_addr),
        .dout0(c1w_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );
    // CONV1 biases (6)
    wire [2:0]       c1b_addr; wire [DATA_W-1:0] c1b_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(6),.ADDR_W(3),
                 .INIT_FILE("hex_weights/conv1_bias.hex")) u_conv1_b_rom (
        .clk(clk),.en(en_conv1_rom),.addr(c1b_addr),
        .dout0(c1b_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );
    // CONV2 weights (2400)
    wire [11:0]      c2w_addr; wire [DATA_W-1:0] c2w_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(2400),.ADDR_W(12),
                 .INIT_FILE("hex_weights/conv2_weights.hex")) u_conv2_w_rom (
        .clk(clk),.en(en_conv2_rom),.addr(c2w_addr),
        .dout0(c2w_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );
    // CONV2 biases (16)
    wire [3:0]       c2b_addr; wire [DATA_W-1:0] c2b_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(16),.ADDR_W(4),
                 .INIT_FILE("hex_weights/conv2_bias.hex")) u_conv2_b_rom (
        .clk(clk),.en(en_conv2_rom),.addr(c2b_addr),
        .dout0(c2b_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );
    // FC1 weights (30720)
    wire [14:0]      f1w_addr; wire [DATA_W-1:0] f1w_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(30720),.ADDR_W(15),
                 .INIT_FILE("hex_weights/fc1_weights.hex")) u_fc1_w_rom (
        .clk(clk),.en(en_fc_rom),.addr(f1w_addr),
        .dout0(f1w_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );
    // FC1 biases (120)
    wire [6:0]       f1b_addr; wire [DATA_W-1:0] f1b_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(120),.ADDR_W(7),
                 .INIT_FILE("hex_weights/fc1_bias.hex")) u_fc1_b_rom (
        .clk(clk),.en(en_fc_rom),.addr(f1b_addr),
        .dout0(f1b_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );
    // FC2 weights (10080)
    wire [13:0]      f2w_addr; wire [DATA_W-1:0] f2w_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(10080),.ADDR_W(14),
                 .INIT_FILE("hex_weights/fc2_weights.hex")) u_fc2_w_rom (
        .clk(clk),.en(en_fc_rom),.addr(f2w_addr),
        .dout0(f2w_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );
    // FC2 biases (84)
    wire [6:0]       f2b_addr; wire [DATA_W-1:0] f2b_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(84),.ADDR_W(7),
                 .INIT_FILE("hex_weights/fc2_bias.hex")) u_fc2_b_rom (
        .clk(clk),.en(en_fc_rom),.addr(f2b_addr),
        .dout0(f2b_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );
    // FC3 weights (840)
    wire [9:0]       f3w_addr; wire [DATA_W-1:0] f3w_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(840),.ADDR_W(10),
                 .INIT_FILE("hex_weights/fc3_weights.hex")) u_fc3_w_rom (
        .clk(clk),.en(en_fc_rom),.addr(f3w_addr),
        .dout0(f3w_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );
    // FC3 biases (10)
    wire [3:0]       f3b_addr; wire [DATA_W-1:0] f3b_d0;
    rom_module #(.DATA_W(DATA_W),.DEPTH(10),.ADDR_W(4),
                 .INIT_FILE("hex_weights/fc3_bias.hex")) u_fc3_b_rom (
        .clk(clk),.en(en_fc_rom),.addr(f3b_addr),
        .dout0(f3b_d0),.dout1(),.dout2(),.dout3(),.dout4()
    );

    // ROM data mux for CONV block
    wire [11:0]      conv_wt_addr_w;
    wire [3:0]       conv_bias_addr_w;
    wire [DATA_W-1:0]conv_wt_data;
    wire [DATA_W-1:0]conv_bias_data;
    assign c1w_addr      = conv_wt_addr_w[7:0];
    assign c2w_addr      = conv_wt_addr_w;
    assign c1b_addr      = conv_bias_addr_w[2:0];
    assign c2b_addr      = conv_bias_addr_w;
    assign conv_wt_data  = en_conv1_rom ? c1w_d0 : c2w_d0;
    assign conv_bias_data= en_conv1_rom ? c1b_d0 : c2b_d0;

    // ROM data mux for FC block
    wire [17:0]      fc_wt_addr_w;
    wire [9:0]       fc_bias_addr_w;
    wire [DATA_W-1:0]fc_wt_data_w;
    wire [DATA_W-1:0]fc_bias_data_w;
    // fc_control outputs global FC address space:
    //   FC1 W: [0..30719], FC2 W: [30720..40799], FC3 W: [40800..41639]
    //   FC1 B: [0..119],   FC2 B: [120..203],    FC3 B: [204..213]
    // Convert to local ROM indices before driving each ROM address port.
    wire [17:0] fc2_w_local = fc_wt_addr_w  - 18'd30720;
    wire [17:0] fc3_w_local = fc_wt_addr_w  - 18'd40800;
    wire [9:0]  fc2_b_local = fc_bias_addr_w - 10'd120;
    wire [9:0]  fc3_b_local = fc_bias_addr_w - 10'd204;
    assign f1w_addr      = fc_wt_addr_w[14:0];
    assign f2w_addr      = fc2_w_local[13:0];
    assign f3w_addr      = fc3_w_local[9:0];
    assign f1b_addr      = fc_bias_addr_w[6:0];
    assign f2b_addr      = fc2_b_local[6:0];
    assign f3b_addr      = fc3_b_local[3:0];
    assign fc_wt_data_w  = (fc_wt_addr_w  < 18'd30720) ? f1w_d0 :
                           (fc_wt_addr_w  < 18'd40800) ? f2w_d0 : f3w_d0;
    assign fc_bias_data_w= (fc_bias_addr_w < 10'd120)  ? f1b_d0 :
                           (fc_bias_addr_w < 10'd204)  ? f2b_d0 : f3b_d0;

    // =========================================================
    //  Master FSM  (paper Fig. 10)
    // =========================================================
    wire        rst_mat1c, rst_mat2c, rst_buff10;
    wire [5:0]  rst_buff2_i;
    wire        en_mxpl1;
    wire [5:0]  en_mxpl2_i;
    wire        en_contcnn, en_add;
    wire [3:0]  conv2_batch, add_count;
    wire        inference_done;
    wire        conv1_done_w, mxpl1_done_w;
    wire        conv2_done_w, mxpl2_done_w;
    wire        add_done_w, fc_done_w;

    assign add_done_w   = 1'b1;   // adder is combinational
    // Reuse the same conv block for both layers, but gate done pulses by
    // active layer to avoid stale CONV1/P00L1 done pulses being consumed as
    // CONV2/POOL2 completion when FSM enters S6/S8.
    assign conv2_done_w = en_conv2_rom & conv1_done_w;
    assign mxpl2_done_w = en_conv2_rom & mxpl1_done_w;

    fsm_control u_fsm (
        .clk(clk),.rst(rst),
        .img_ready   (img_ready_safe),
        .conv1_done  (conv1_done_w), .mxpl1_done(mxpl1_done_w),
        .conv2_done  (conv2_done_w), .mxpl2_done(mxpl2_done_w),
        .add_done    (add_done_w),   .fc_done   (fc_done_w),
        .rst_mat1c   (rst_mat1c),    .rst_mat2c (rst_mat2c),
        .rst_buff10  (rst_buff10),   .rst_buff2_i(rst_buff2_i),
        .en_mxpl1    (en_mxpl1),     .en_mxpl2_i(en_mxpl2_i),
        .en_contcnn  (en_contcnn),   .en_add    (en_add),
        .en_conv1_rom(en_conv1_rom), .en_conv2_rom(en_conv2_rom),
        .en_fc_rom   (en_fc_rom),
        .conv2_batch (conv2_batch),  .add_count (add_count),
        .inference_done(inference_done)
    );

    // =========================================================
    //  Conv+MaxPool Block
    // =========================================================
    wire [15:0]      conv_feat_rd_addr;
    wire [11:0]      conv_feat_wr_addr;
    wire [DATA_W-1:0]conv_feat_wr_data;
    wire             conv_feat_wr_en;
    wire [11:0]      conv_mxpl_wr_addr;
    wire [DATA_W-1:0]conv_mxpl_wr_data;
    wire             conv_mxpl_wr_en;
    wire             conv_pool_rd_active;

    wire [12:0] conv_in_base  = en_conv1_rom ? BASE_IMG   : BASE_POOL1;
    wire [12:0] conv_out_base = en_conv1_rom ? BASE_CONV1 : BASE_CONV2;
    wire [12:0] pool_out_base = en_conv1_rom ? BASE_POOL1 : BASE_POOL2;


    // Runtime config: CONV1=6 filters 28x28 1ch, CONV2=16 filters 12x12 6ch
    wire [4:0] cfg_n_filt = en_conv1_rom ? 5'd6  : 5'd16;
    wire [4:0] cfg_in_h   = en_conv1_rom ? 5'd28 : 5'd12;
    wire [4:0] cfg_in_w   = en_conv1_rom ? 5'd28 : 5'd12;
    wire [2:0] cfg_in_ch  = en_conv1_rom ? 3'd1  : 3'd6;

    conv_maxpool_block #(
        .DATA_W(DATA_W),.ACC_W(ACC_W),
        .FILT(5),.FRAC_BITS(FRAC_BITS)
    ) u_conv (
        .n_filt      (cfg_n_filt),
        .in_h        (cfg_in_h),
        .in_w        (cfg_in_w),
        .in_ch       (cfg_in_ch),
        .clk         (clk),.rst(rst),
        .rst_matc    (en_conv1_rom ? rst_mat1c  : rst_mat2c),
        .rst_buff    (en_conv1_rom ? rst_buff10 : rst_buff2_i[conv2_batch]),
        .en_mxpl     (en_conv1_rom ? en_mxpl1  : |en_mxpl2_i),
        .start       (en_conv1_rom | en_conv2_rom),
        .feat_rd_addr(conv_feat_rd_addr),
        .feat_rd_data(bram_b_dout),
        .wt_rd_addr  (conv_wt_addr_w),
        .wt_rd_data  (conv_wt_data),
        .bias_addr   (conv_bias_addr_w),
        .bias_data   (conv_bias_data),
        .feat_wr_addr(conv_feat_wr_addr),
        .feat_wr_data(conv_feat_wr_data),
        .feat_wr_en  (conv_feat_wr_en),
        .mxpl_wr_addr(conv_mxpl_wr_addr),
        .mxpl_wr_data(conv_mxpl_wr_data),
        .mxpl_wr_en  (conv_mxpl_wr_en),
        .conv_done   (conv1_done_w),
        .mxpl_done   (mxpl1_done_w),
        .pool_rd_active(conv_pool_rd_active)
    );

    // =========================================================
    //  MAC Unit  (paper Fig. 5a)
    // =========================================================
    wire              mac_bias_en_w, mac_load_acc_w;
    wire signed [ACC_W-1:0]  mac_out_w;
    wire signed [DATA_W-1:0] mac_act_out_w;

    mac_unit #(.DATA_W(DATA_W),.ACC_W(ACC_W),.FRAC_BITS(FRAC_BITS)) u_mac (
        .clk(clk),.rst(rst),
        .FV      (bram_b_dout),
        .W       (fc_wt_data_w),
        .bias    (fc_bias_data_w),
        .bias_en (mac_bias_en_w),
        .load_acc(mac_load_acc_w),
        .mac_out (mac_out_w),
        .act_out (mac_act_out_w)
    );

    // =========================================================
    //  FC Control  (paper Fig. 7)
    // =========================================================
    wire [12:0]      fc_rd_addr_w;
    wire [9:0]       fc_out_wr_addr_w;
    wire [DATA_W-1:0]fc_out_wr_data_w;
    wire             fc_out_wr_en_w;
    wire [12:0]      fc_fv_wr_addr_w;
    wire [DATA_W-1:0]fc_fv_wr_data_w;
    wire             fc_fv_wr_en_w;

    fc_control #(
        .DATA_W(DATA_W),.ACC_W(ACC_W),
        .N_FC1_IN(256),.N_FC1_OUT(120),
        .N_FC2_IN(120),.N_FC2_OUT(84),
        .N_FC3_IN(84), .N_FC3_OUT(10),
        .FRAC_BITS(FRAC_BITS)
    ) u_fc (
        .clk         (clk),.rst(rst),
        .en_contcnn  (en_contcnn),
        .fv_rd_addr  (fc_rd_addr_w),
        .fv_rd_data  (bram_b_dout),
        .fv_rd_en    (en_fc_rom),
        .wt_rd_addr  (fc_wt_addr_w),
        .wt_rd_data  (fc_wt_data_w),
        .bias_rd_addr(fc_bias_addr_w),
        .bias_rd_data(fc_bias_data_w),
        .out_wr_addr (fc_out_wr_addr_w),
        .out_wr_data (fc_out_wr_data_w),
        .out_wr_en   (fc_out_wr_en_w),
        .fv_wr_addr  (fc_fv_wr_addr_w),
        .fv_wr_data  (fc_fv_wr_data_w),
        .fv_wr_en    (fc_fv_wr_en_w),
        .bias_en     (mac_bias_en_w),
        .load_acc    (mac_load_acc_w),
        .act_out     (mac_act_out_w),
        .fc_done     (fc_done_w)
    );

    // =========================================================
    //  BRAM Port B read address MUX (single always @(*))
    // =========================================================
    localparam [2:0] AX_IDLE=3'd0, AX_ADDR=3'd1, AX_READ=3'd2,
                     AX_COMP=3'd3, AX_DONE=3'd4;
    reg [2:0]  ax_st;
    reg [3:0]  ax_idx;
    reg [12:0] ax_addr;

    always @(*) begin
        if (en_conv1_rom | en_conv2_rom) begin
            if (conv_pool_rd_active)
                bram_b_addr = conv_out_base + {1'b0, conv_feat_rd_addr[11:0]};
            else
                bram_b_addr = conv_in_base  + {1'b0, conv_feat_rd_addr[11:0]};
        end
        else if (en_fc_rom)
            bram_b_addr = fc_rd_addr_w;
        else if (ax_st == AX_ADDR || ax_st == AX_READ)
            bram_b_addr = ax_addr;
        else
            bram_b_addr = 13'd0;
    end

    // BRAM Port A write MUX (single always @(*))
    always @(*) begin
        if (en_conv1_rom | en_conv2_rom) begin
            if (conv_mxpl_wr_en) begin
                layer_wr_addr = pool_out_base + {1'b0,conv_mxpl_wr_addr};
                layer_wr_data = conv_mxpl_wr_data;
                layer_wr_en   = 1'b1;
            end else begin
                layer_wr_addr = conv_out_base + {1'b0,conv_feat_wr_addr};
                layer_wr_data = conv_feat_wr_data;
                layer_wr_en   = conv_feat_wr_en;
            end
        end else if (en_fc_rom) begin
            if (fc_out_wr_en_w) begin
                layer_wr_addr = BASE_FC3 + {3'b0,fc_out_wr_addr_w};
                layer_wr_data = fc_out_wr_data_w;
                layer_wr_en   = 1'b1;
            end else if (fc_fv_wr_en_w) begin
                layer_wr_addr = fc_fv_wr_addr_w;
                layer_wr_data = fc_fv_wr_data_w;
                layer_wr_en   = 1'b1;
            end else begin
                layer_wr_addr = 13'd0;
                layer_wr_data = {DATA_W{1'b0}};
                layer_wr_en   = 1'b0;
            end
        end else begin
            layer_wr_addr = 13'd0;
            layer_wr_data = {DATA_W{1'b0}};
            layer_wr_en   = 1'b0;
        end
    end

    // =========================================================
    //  Argmax comparator  (paper Sec. 3.1)
    // =========================================================
    reg [DATA_W-1:0] fc3_out [0:9];
    reg [3:0]  pred_class;
    reg        argmax_done;

    // Sequential argmax: compare one candidate per clock
    reg [3:0]  cmp_idx;    // current comparison index
    reg [3:0]  best_idx;   // best so far

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            ax_st<=AX_IDLE; ax_idx<=0; ax_addr<=BASE_FC3;
            pred_class<=0; argmax_done<=0;
            cmp_idx<=0; best_idx<=0;
        end else begin
            argmax_done <= 0;
            case(ax_st)
                AX_IDLE: begin
                    if (inference_done) begin
                        ax_idx  <= 0;
                        ax_addr <= BASE_FC3;   // address for index 0
                        ax_st   <= AX_ADDR;    // wait 1 cycle for BRAM latency
                    end
                end

                // Hold address; doutb will be valid next cycle
                AX_ADDR: begin
                    ax_st <= AX_READ;
                end

                // doutb is now valid for ax_addr set in AX_ADDR / previous AX_READ
                AX_READ: begin
                    fc3_out[ax_idx] <= bram_b_dout;   // sample valid data
                    if (ax_idx == 4'd9) begin
                        // All 10 logits captured; start comparison
                        ax_st    <= AX_COMP;
                        ax_idx   <= 0;
                        best_idx <= 0;
                        cmp_idx  <= 1;
                    end else begin
                        ax_idx  <= ax_idx + 1;
                        ax_addr <= BASE_FC3 + {9'b0,ax_idx} + 1; // next address
                        ax_st   <= AX_ADDR;  // wait 1 cycle for next address
                    end
                end

                // Sequential comparison: one candidate per clock
                AX_COMP: begin
                    if (cmp_idx == 4'd10) begin
                        pred_class <= best_idx;
                        ax_st      <= AX_DONE;
                    end else begin
                        if ($signed(fc3_out[cmp_idx]) > $signed(fc3_out[best_idx]))
                            best_idx <= cmp_idx;
                        cmp_idx <= cmp_idx + 1;
                    end
                end

                AX_DONE: begin
                    argmax_done <= 1;
                    ax_st       <= AX_IDLE;
                end

                default: ax_st <= AX_IDLE;
            endcase
        end
    end

    // =========================================================
    //  UART TX + LED output
    // =========================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin tx_data_r<=8'h30; tx_start_r<=0; end
        else begin
            tx_start_r <= 0;
            if (argmax_done && !tx_busy) begin
                tx_data_r  <= 8'h30 + {4'b0,pred_class}; // ASCII '0'..'9'
                tx_start_r <= 1;
            end
        end
    end

    assign led[3:0] = pred_class;
    assign led[7:4] = 4'b0;
    assign led[8]   = img_receiving;
    assign led[9]   = argmax_done;

endmodule
