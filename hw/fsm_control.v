

module fsm_control (
    input  wire        clk, rst,
    input  wire        img_ready,
    input  wire        conv1_done, mxpl1_done,
    input  wire        conv2_done, mxpl2_done,
    input  wire        add_done,   fc_done,
    output reg         rst_mat1c,  rst_mat2c,
    output reg         rst_buff10,
    output reg  [5:0]  rst_buff2_i,
    output reg         en_mxpl1,
    output reg  [5:0]  en_mxpl2_i,
    output reg         en_contcnn, en_add,
    output reg         en_conv1_rom, en_conv2_rom, en_fc_rom,
    output reg  [3:0]  conv2_batch, add_count,
    output reg         inference_done
);
    localparam S0=4'd0, S1=4'd1, S2=4'd2, S3=4'd3,
               S4=4'd4, S5=4'd5, S6=4'd6, S7=4'd7,
               S8=4'd8, S9=4'd9, S10=4'd10, S11=4'd11;

    reg [3:0] state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state<=S0; rst_mat1c<=1; rst_mat2c<=1;
            rst_buff10<=1; rst_buff2_i<=6'h3F;
            en_mxpl1<=0; en_mxpl2_i<=0;
            en_contcnn<=0; en_add<=0;
            en_conv1_rom<=0; en_conv2_rom<=0; en_fc_rom<=0;
            conv2_batch<=0; add_count<=0; inference_done<=0;
        end else begin
            en_mxpl1<=0; en_mxpl2_i<=0; en_contcnn<=0;
            en_add<=0; inference_done<=0;

            case(state)
                S0: begin
                    rst_mat1c<=1; rst_mat2c<=1; rst_buff10<=1;
                    rst_buff2_i<=6'h3F; en_conv1_rom<=0;
                    en_conv2_rom<=0; en_fc_rom<=0;
                    conv2_batch<=0; add_count<=0;
                    if(img_ready) begin en_conv1_rom<=1; state<=S1; end
                end
                S1: begin rst_mat1c<=0; state<=S2; end
                S2: begin
                    rst_buff10<=0;
                    en_mxpl1<=1;
                    if(conv1_done) state<=S3;
                end
                S3: begin
                    rst_mat1c<=1;
                    en_mxpl1<=0;   // pool already started; de-assert
                    state<=S4;
                end
                S4: begin
                    rst_buff10<=1;
                    if(mxpl1_done) state<=S5;
                end
                S5: begin
                    en_conv1_rom<=0; en_conv2_rom<=1;
                    rst_mat2c<=0; state<=S6;
                end
                S6: begin
                    rst_buff2_i[conv2_batch]<=0;
                    en_mxpl2_i[conv2_batch]<=1;
                    if(conv2_done) state<=S7;
                end
                S7: begin
                    rst_mat2c<=1;
                    en_mxpl2_i[conv2_batch]<=0;
                    state<=S8;
                end
                S8: begin
                    rst_buff2_i[conv2_batch]<=1;
                    en_add<=1;
                    if(mxpl2_done) begin
                        if(conv2_batch==4'd5) begin
                            conv2_batch<=0; state<=S9;
                        end else begin
                            conv2_batch<=conv2_batch+1;
                            rst_mat2c<=0; state<=S5;
                        end
                    end
                end
                S9: begin
                    en_add<=1; add_count<=add_count+1;
                    if(add_done) state<=S10;
                end
                S10: begin
                    if(add_count==4'd15) begin
                        add_count<=0; state<=S11;
                    end else state<=S9;
                end
                S11: begin
                    en_conv2_rom<=0; en_fc_rom<=1;
                    en_contcnn<=1;
                    if(fc_done) begin
                        en_fc_rom<=0; inference_done<=1; state<=S0;
                    end
                end
                default: state<=S0;
            endcase
        end
    end
endmodule
