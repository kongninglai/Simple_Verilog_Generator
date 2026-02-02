# Simple Verilog Generator

## Usage

`python3 generate_comb.py path_to_comb_input --module=module_name`
`python3 generate_fsm.py path_to_fsm_input --module=module_name`
For example, 
`python3 generate.py examples/xor3.txt --module=xor3`

The output verilog code would be written to `{module}.v`.
