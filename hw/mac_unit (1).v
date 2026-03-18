// ============================================================
//  mac_unit.v  —  Multiply-Accumulate Unit for FC Layers
//  Paper reference : Mukhopadhyay et al., Fig. 5(a)
//  Target          : ZedBoard XC7Z020, Vivado 2019.1
//
//  Structure EXACTLY as in paper Fig. 5(a):
//    FV  ──┐
//    W   ──┴─► [Multiplier] ──► [Register m_out] ──► [Adder] ──►
//                                                         ▲
//    bias ────────────────────────────────────────────────┘
//                                                         │
//                                                    [MUX Vdd/0]
//                                                         │
//    bias_en ─────────────────────────────────────────────►(MUX sel)
//    load_acc ────────────────────────────────────────────►(CE)
//                                                         │
//                                                    [Register mac_out]
//                                                         │
//                                                    mac_out[MSB]──►[MUX]──►act_out
//                                                                       ▲
//                                                                     ReLU──►1
//
//  Signals (paper Fig. 5a):
//    bias_en  : load bias value into accumulator at start of each neuron
//    load_acc : CE (clock enable) for output register; enables accumulation
//
//  Pipelining (paper Sec 3.1.1):
//    "registers are used after the multiply and add stages.
//     Pipelining will help in reducing the critical path delay
//     thereby increasing the throughput of the MAC unit."
// ============================================================

module mac_unit #(
    parameter DATA_W    = 8,
    parameter ACC_W     = 24,
    parameter FRAC_BITS = 6
)(
    input  wire                       clk,
    input  wire                       rst,

    // ── Data inputs ──────────────────────────────────────
    input  wire signed [DATA_W-1:0]   FV,         // feature vector input
    input  wire signed [DATA_W-1:0]   W,          // weight
    input  wire signed [DATA_W-1:0]   bias,       // bias for this neuron

    // ── Control signals (paper Fig. 5a) ──────────────────
    input  wire                       bias_en,    // load bias to accumulator
    input  wire                       load_acc,   // enable accumulation (CE)

    // ── Outputs ──────────────────────────────────────────
    output wire signed [ACC_W-1:0]    mac_out,    // raw accumulator output
    output wire signed [DATA_W-1:0]   act_out     // after ReLU (paper Fig 5b)
);

    // ── Pipeline register 1: multiply stage ──────────────
    reg signed [2*DATA_W-1:0] m_out;   // paper names this "m_out"
    always @(posedge clk or posedge rst) begin
        if (rst) m_out <= 0;
        else     m_out <= FV * W;
    end

    // Sign-extend m_out to ACC_W
    wire signed [ACC_W-1:0] m_out_ext;
    assign m_out_ext = {{(ACC_W-2*DATA_W){m_out[2*DATA_W-1]}}, m_out};

    // Sign-extend bias to ACC_W
    wire signed [ACC_W-1:0] bias_ext;
    assign bias_ext = {{(ACC_W-DATA_W){bias[DATA_W-1]}}, bias} <<< FRAC_BITS;

    // ── MUX (paper: MUX with Vdd/0 on one input) ─────────
    // When bias_en=1 : adder input from bias (initialise acc)
    // When bias_en=0 : adder input from m_out (accumulate)
    wire signed [ACC_W-1:0] adder_b;
    assign adder_b = bias_en ? bias_ext : m_out_ext;

    // ── Adder ─────────────────────────────────────────────
    wire signed [ACC_W-1:0] adder_out;
    // When bias_en=1: adder_out = 0 + bias_ext (clear acc, load bias)
    // When bias_en=0: adder_out = mac_out_reg + m_out_ext
    wire signed [ACC_W-1:0] mac_out_reg_w; // forward declaration
    assign adder_out = (bias_en ? {ACC_W{1'b0}} : mac_out_reg_w) + adder_b;

    // ── Pipeline register 2: accumulator (with CE) ────────
    reg signed [ACC_W-1:0] mac_out_reg;
    assign mac_out_reg_w = mac_out_reg;

    always @(posedge clk or posedge rst) begin
        if (rst)
            mac_out_reg <= 0;
        else if (bias_en || load_acc)
            mac_out_reg <= adder_out;
        // else: hold (CE=0)
    end

    assign mac_out = mac_out_reg;

    // ── ReLU activation (paper Fig. 5b) ───────────────────
    // "mac_out[MSB] = 1 → output 0, else output mac_out"
    wire sign_bit = mac_out_reg[ACC_W-1];
    wire overflow = |mac_out_reg[ACC_W-2:DATA_W+FRAC_BITS-1];

    assign act_out = sign_bit  ? {DATA_W{1'b0}} :
                     overflow  ? {1'b0, {(DATA_W-1){1'b1}}} :
                                 mac_out_reg[DATA_W+FRAC_BITS-2:FRAC_BITS];

endmodule
