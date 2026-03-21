import re
import sys
import os

try:
    import trace_layers
except ImportError:
    print("ERROR: trace_layers.py not found or failed to execute.")
    sys.exit(1)

def parse_log_and_compare(log_path):
    if not os.path.exists(log_path):
        print(f"ERROR: Cannot find {log_path}")
        return

    print(f"Parsing RTL Simulation Log: {log_path}...\n{'='*50}")
    
    conv2_wr_regex = re.compile(r"\[CONV2_WR\s+#(\d+)\]\s+filter=(\d+)\s+acc_before_relu=(-?\d+)\s+data=(\d+)")
    fc1_out_regex  = re.compile(r"FC1_OUT:\s+neuron=(\d+)\s+act=(-?\d+)")
    fc3_out_regex  = re.compile(r"FC3_OUT:\s+neuron=(\d+)\s+logit=(-?\d+)")
    
    conv2_errors = 0
    fc1_errors = 0
    fc3_errors = 0

    with open(log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            
            m = conv2_wr_regex.search(line)
            if m:
                write_idx = int(m.group(1))
                filt = int(m.group(2))
                rtl_acc = int(m.group(3))
                rtl_data = int(m.group(4))
                
                if write_idx < len(trace_layers.conv2_out):
                    py_data = trace_layers.conv2_out[write_idx]
                    if rtl_data != py_data:
                        print(f"[Line {line_num}] CONV2 MISMATCH @ write={write_idx} (filter={filt}): RTL output {rtl_data} != Python {py_data} (RTL acc_before_clip={rtl_acc})")
                        conv2_errors += 1
                        if conv2_errors > 20: 
                            conv2_errors = -999 

            m = fc1_out_regex.search(line)
            if m:
                neuron = int(m.group(1))
                rtl_act = int(m.group(2))
                if neuron < len(trace_layers.fc1_out):
                    py_act = trace_layers.fc1_out[neuron]
                    if rtl_act != py_act:
                        print(f"[Line {line_num}] FC1 MISMATCH @ neuron={neuron}: RTL output {rtl_act} != Python {py_act}")
                        fc1_errors += 1
                        
            m = fc3_out_regex.search(line)
            if m:
                neuron = int(m.group(1))
                rtl_logit = int(m.group(2))
                if neuron < len(trace_layers.fc3_out):
                    py_logit = trace_layers.fc3_out[neuron]
                    if rtl_logit != py_logit:
                        print(f"[Line {line_num}] FC3 (Logit) MISMATCH @ neuron={neuron}: RTL output {rtl_logit} != Python {py_logit}")
                        fc3_errors += 1

    print(f"\n{'='*50}")
    print(f"Verification Results:")
    print(f"CONV2: {'PASS' if conv2_errors == 0 else f'FAIL ({max(0, conv2_errors)} errors)'}")
    print(f"FC1  : {'PASS' if fc1_errors == 0 else f'FAIL ({fc1_errors} errors)'}")
    print(f"FC3  : {'PASS' if fc3_errors == 0 else f'FAIL ({fc3_errors} errors)'}")
    
    if conv2_errors == 0 and fc1_errors == 0 and fc3_errors == 0:
        print("\nSUCCESS! RTL Simulation PERFECTLY matches Python Golden Model.")
    else:
        print("\nFAILURE. Check the exact mismatched indexes above to trace the pipeline bug.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_log_and_compare(sys.argv[1])
    else:
        parse_log_and_compare("results.log")
