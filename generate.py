#!/usr/bin/env python3
from __future__ import annotations
"""
Run Espresso on a PLA/truth-table file and derive optimized boolean expressions.

Usage:
  python3 run_espresso.py temp.in
  python3 run_espresso.py temp.in --espresso ./espresso --out temp.out

What it does:
  1) Calls:  ./espresso <input> > <output>
  2) Parses the PLA-style output (like your expected format)
  3) Converts cubes into SOP boolean expressions for each output bit

Notes:
  - Assumes AND/OR/NOT logic with don't-care '-' in input cubes.
  - For output cubes, only '1' contributes a product term to that output's SOP.
    '0' and '-' do not add a term.
"""

"""
lib1
====

module  nand2$(out, in0, in1);
module  nand3$(out, in0, in1, in2);
module  nand4$(out, in0, in1, in2, in3);
module  and2$(out, in0, in1);
module  and3$(out, in0, in1, in2);
module  and4$(out, in0, in1, in2, in3);
module  nor2$(out, in0, in1);
module  nor3$(out, in0, in1, in2);
module  nor4$(out, in0, in1, in2, in3);
module  or2$(out, in0, in1);
module  or3$(out, in0, in1, in2);
module  or4$(out, in0, in1, in2, in3);
module  xor2$ (out, in0, in1);
module  xnor2$ (out, in0, in1);
module  inv1$ (out, in);
module  dff$(clk, d, q, qbar, r, s);
"""


import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
SCRIPT_DIR = Path(__file__).resolve().parent


DEBUG = 0

@dataclass
class PLA:
    n_inputs: int
    n_outputs: int
    input_names: List[str]
    output_names: List[str]
    cubes: List[Tuple[str, str]]  # (input_pattern, output_pattern)
    raw_lines: List[str]



gate_and2 = "and2$ {inst_name}({out}, {in0}, {in1});"
gate_and3 = "and3$ {inst_name}({out}, {in0}, {in1}, {in2});"
gate_and4 = "and4$ {inst_name}({out}, {in0}, {in1}, {in2}, {in3});"

gate_or2 = "or2$ {inst_name}({out}, {in0}, {in1});"
gate_or3 = "or3$ {inst_name}({out}, {in0}, {in1}, {in2});"
gate_or4 = "or4$ {inst_name}({out}, {in0}, {in1}, {in2}, {in3});"

gate_nand = "nand2$ {inst_name}({out}, {in0}, {in1});"
gate_inv = "inv1$ {inst_name}({out}, {in0});"

@dataclass
class GATES:
    gate_type: str
    n_gates: int
    input_list: List[str]
    input_idx: int
    output_list: List[str]
    wire_list: List[str]
    
def f(x, gates:GATES, output_name, inst_name):
    if (gates.gate_type == "or"):
        gate2 = gate_or2
        gate3 = gate_or3
        gate4 = gate_or4
    else:
        gate2 = gate_and2
        gate3 = gate_and3
        gate4 = gate_and4
    if x == 1:
        return "1"
    elif x == 2:
        in0 = gates.input_list[gates.input_idx]
        in1 = gates.input_list[gates.input_idx+1]
        gates.input_idx = gates.input_idx + 2
        gates.output_list.append(gate2.format(inst_name=inst_name, out=output_name, in0=in0, in1=in1))
        return "gate2(1, 1)"
    elif x == 3:
        in0 = gates.input_list[gates.input_idx]
        in1 = gates.input_list[gates.input_idx+1]
        in2 = gates.input_list[gates.input_idx+2]
        gates.input_idx = gates.input_idx + 3
        gates.output_list.append(gate3.format(inst_name=inst_name, out=output_name, in0=in0, in1=in1, in2=in2))
        return "gate3(1, 1, 1)"
    elif x == 4:
        in0 = gates.input_list[gates.input_idx]
        in1 = gates.input_list[gates.input_idx+1]
        in2 = gates.input_list[gates.input_idx+2]
        in3 = gates.input_list[gates.input_idx+3]
        gates.input_idx = gates.input_idx + 4
        gates.output_list.append(gate4.format(inst_name=inst_name, out=output_name, in0=in0, in1=in1, in2=in2, in3=in3))
        return "gate4(1, 1, 1, 1)"
    else:
        inst_prefix = gates.gate_type
        if x <= 7:
            gates.n_gates = gates.n_gates + 1
            inst0_name = f"{inst_prefix}inst{gates.n_gates}"
            wire_name = f"{gates.gate_type}_{gates.n_gates}_o"
            in1 = gates.input_list[gates.input_idx]
            in2 = gates.input_list[gates.input_idx+1]
            in3 = gates.input_list[gates.input_idx+2]
            gates.input_idx = gates.input_idx + 3
            gates.output_list.append(gate4.format(inst_name=inst_name, out=output_name, in0=wire_name, in1=in1, in2=in2, in3=in3))
            gates.wire_list.append(wire_name)

            return f"gate4({f(x-3, gates, wire_name, inst0_name)}, 1, 1, 1)"
        elif x > 7 and x <= 10:
            gates.n_gates = gates.n_gates + 1
            inst0_name = f"{inst_prefix}inst{gates.n_gates}"
            wire0_name = f"{gates.gate_type}_{gates.n_gates}_o"
            gates.n_gates = gates.n_gates + 1
            inst1_name = f"{inst_prefix}inst{gates.n_gates}"
            wire1_name = f"{gates.gate_type}_{gates.n_gates}_o"
            in2 = gates.input_list[gates.input_idx]
            in3 = gates.input_list[gates.input_idx+1]
            gates.input_idx = gates.input_idx + 2
            gates.output_list.append(gate4.format(inst_name=inst_name, out=output_name, in0=wire0_name, in1=wire1_name, in2=in2, in3=in3))
            gates.wire_list.append(wire0_name)
            gates.wire_list.append(wire1_name)
            return f"gate4({f(4, gates, wire0_name, inst0_name)}, {f(x-6, gates, wire1_name, inst1_name)}, 1, 1)"
        elif x >10 and x <= 13:
            gates.n_gates = gates.n_gates + 1
            inst0_name = f"{inst_prefix}inst{gates.n_gates}"
            wire0_name = f"{gates.gate_type}_{gates.n_gates}_o"
            gates.n_gates = gates.n_gates + 1
            inst1_name = f"{inst_prefix}inst{gates.n_gates}"
            wire1_name = f"{gates.gate_type}_{gates.n_gates}_o"
            gates.n_gates = gates.n_gates + 1
            inst2_name = f"{inst_prefix}inst{gates.n_gates}"
            wire2_name = f"{gates.gate_type}_{gates.n_gates}_o"
            in3 = gates.input_list[gates.input_idx]
            gates.input_idx = gates.input_idx + 1
            gates.output_list.append(gate4.format(inst_name=inst_name, out=output_name, in0=wire0_name, in1=wire1_name, in2=wire2_name, in3=in3))
            gates.wire_list.append(wire0_name)
            gates.wire_list.append(wire1_name)
            gates.wire_list.append(wire2_name)
            return f"gate4({f(4, gates, wire0_name, inst0_name)}, {f(4, gates, wire1_name, inst1_name)}, {f(x-9, gates, wire2_name, inst2_name)}, 1)"
        else:
            gates.n_gates = gates.n_gates + 1
            inst0_name = f"{inst_prefix}inst{gates.n_gates}"
            wire0_name = f"{gates.gate_type}_{gates.n_gates}_o"
            gates.n_gates = gates.n_gates + 1
            inst1_name = f"{inst_prefix}inst{gates.n_gates}"
            wire1_name = f"{gates.gate_type}_{gates.n_gates}_o"
            gates.n_gates = gates.n_gates + 1
            inst2_name = f"{inst_prefix}inst{gates.n_gates}"
            wire2_name = f"{gates.gate_type}_{gates.n_gates}_o"
            gates.n_gates = gates.n_gates + 1
            inst3_name = f"{inst_prefix}inst{gates.n_gates}"
            wire3_name = f"{gates.gate_type}_{gates.n_gates}_o"
            gates.output_list.append(gate4.format(inst_name=inst_name, out=output_name, in0=wire0_name, in1=wire1_name, in2=wire2_name, in3=wire3_name))
            gates.wire_list.append(wire0_name)
            gates.wire_list.append(wire1_name)
            gates.wire_list.append(wire2_name)
            gates.wire_list.append(wire3_name)
            return f"gate4({f(4, gates, wire0_name, inst0_name)}, {f(4, gates, wire1_name, inst1_name)}, {f(4, gates, wire2_name, inst2_name)}, {f(x-12, gates, wire3_name, inst3_name)})"
   
def run_espresso(in_path: Path, out_path: Path, espresso_path: Path) -> None:
    if not espresso_path.exists():
        raise FileNotFoundError(f"Espresso not found at: {espresso_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Equivalent to: ./espresso temp.in > temp.out
    result = subprocess.run(
        [str(espresso_path), str(in_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Espresso failed.\n"
            f"Return code: {result.returncode}\n"
            f"STDERR:\n{result.stderr}"
        )

    out_path.write_text(result.stdout, encoding="utf-8")


def parse_pla(out_path: Path) -> PLA:
    lines = out_path.read_text(encoding="utf-8", errors="replace").splitlines()

    n_inputs = None
    n_outputs = None
    input_names = []
    output_names = []
    cubes = []

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        tokens = s.split()
        cmd = tokens[0]

        if cmd == ".i":
            if len(tokens) != 2:
                raise ValueError(f"Bad .i line: {line}")
            n_inputs = int(tokens[1])

        elif cmd == ".o":
            if len(tokens) != 2:
                raise ValueError(f"Bad .o line: {line}")
            n_outputs = int(tokens[1])

        elif cmd == ".ilb":
            input_names = tokens[1:]

        elif cmd == ".ob":
            output_names = tokens[1:]

        elif cmd == ".p":
            # product count is optional for parsing
            pass

        elif cmd == ".e":
            break

        elif cmd.startswith("."):
            # ignore other directives safely
            pass

        else:
            # cube line: "<inputs> <outputs>"
            if len(tokens) != 2:
                raise ValueError(f"Bad cube line: {line}")
            cubes.append((tokens[0], tokens[1]))

    if n_inputs is None or n_outputs is None:
        raise ValueError("Missing .i or .o in PLA")

    if not input_names:
        input_names = [f"in{i}" for i in range(n_inputs)]
    if not output_names:
        output_names = [f"out{i}" for i in range(n_outputs)]

    if len(input_names) != n_inputs:
        raise ValueError("Mismatch between .i and .ilb")

    if len(output_names) != n_outputs:
        raise ValueError("Mismatch between .o and .ob")

    return PLA(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        input_names=input_names,
        output_names=output_names,
        cubes=cubes,
        raw_lines=lines,
    )

def get_wire_code(wire_list: List[str]):
    wire_code = "wire "
    for i in range(len(wire_list)-1):
        wire_code = wire_code + wire_list[i] + " ,"
    wire_code = wire_code + wire_list[i+1] + ";"
    return wire_code

def debug_print(str):
    if DEBUG:
        print(str)
    else:
        pass
def debug_print_list(info, printlist):
    if DEBUG:
        print(f"@{info}:")
        for str in printlist:
            print(str)
    else:
        pass

def get_and_to_or_name(idx):
    return f"and_{idx}_o"

# module  inv1$ (out, in);
def cube_to_gate(pla):
    pat_dict = {name: [] for name in pla.output_names}
    idx = 0

    wire_code_list = []
    gate_code_list = []

    inv_wire_list = []
    inv_code_list = []
    tot_inv = 0
 
    tot_or = 0
    tot_and = len(pla.cubes)
    and_code_start = 0
    and_o_wire_list = []
    for inp_pat, out_pat in pla.cubes:

        for i in range(pla.n_inputs):
                if inp_pat[i] == "0":
                    inv_i = pla.input_names[i]
                    inv_o = f"{inv_i.replace('[', '').replace(']', '')}_inv"
                    if not (inv_o in inv_wire_list):
                        inv_inst = f"invinst{tot_inv}"
                        inv_wire_list.append(inv_o)
                        inv_code_list.append(gate_inv.format(inst_name=inv_inst, out=inv_o, in0=inv_i))
                        tot_inv += 1
        
        for i in range(pla.n_outputs):
            if out_pat[i] == '1':
                pat_dict[pla.output_names[i]].append(idx)
        idx += 1
    if len(inv_wire_list) > 0:
            wire_code_list.append(get_wire_code(inv_wire_list))
            gate_code_list.extend(inv_code_list)
            and_code_start = len(gate_code_list)

    idx = 0
    for inp_pat, out_pat in pla.cubes:
        and_input_list = []
        for i in range(pla.n_inputs):
                i_name = pla.input_names[i]
                if inp_pat[i] == "0":
                    and_input_list.append(f"{i_name.replace('[', '').replace(']', '')}_inv")
                elif inp_pat[i] == "1":
                    and_input_list.append(i_name)

        andgates = GATES(
            gate_type="and",
            n_gates = tot_and,
            input_list = and_input_list,
            input_idx = 0,
            output_list = [],
            wire_list = []
        )
        and_o = get_and_to_or_name(idx)
        and_inst = f"andinst{idx}"
        _ = f(len(and_input_list), andgates, and_o, and_inst)
        tot_and = andgates.n_gates + 1
        gate_code_list.extend(reversed(andgates.output_list))
        if len(andgates.wire_list) > 0:
            wire_code_list.append(get_wire_code(andgates.wire_list))

        and_o_wire_list.append(and_o)
        idx += 1   

    
    for name in pla.output_names:
        or_input_list = []
        or_o = name
        or_inst = f"orinst{tot_or}"
        for idx in pat_dict[name]:
            or_input_list.append(get_and_to_or_name(idx))
        num_or = len(or_input_list)
        if num_or > 1:
            orgates = GATES(
                    gate_type="or",
                    n_gates = tot_or,
                    input_list = or_input_list,
                    input_idx = 0,
                    output_list = [],
                    wire_list = []
            )

            _ = f(num_or, orgates, or_o, or_inst)
            tot_or = orgates.n_gates + 1
            gate_code_list.extend(reversed(orgates.output_list))
            if len(orgates.wire_list) > 0:
                wire_code_list.append(get_wire_code(orgates.wire_list))
        else:
            idx = pat_dict[name][0]
            and_o_temp = get_and_to_or_name(idx)
            gate_code_list[idx+and_code_start] = gate_code_list[idx+and_code_start].replace(and_o_temp, name)
            if and_o_temp in and_o_wire_list:
                and_o_wire_list.remove(and_o_temp)
    if len(and_o_wire_list) > 0:      
        wire_code_list.append(get_wire_code(and_o_wire_list))
    wire_code_list.extend(gate_code_list)
    return wire_code_list
    # for name in pla.output_names:
    #     print(pat_dict[name])
    


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Input PLA/truth-table file (e.g., temp.in)")
    ap.add_argument("--module", default="module", help="module name")
    # ap.add_argument("--espresso_out", type=Path, default=Path("temp.out"), help="Output file (espresso stdout)")
    # ap.add_argument("--not-style", choices=["!", "~"], default="!", help="NOT operator style in expressions")
    # ap.add_argument("--print-pla", action="store_true", help="Also print the parsed PLA cubes")
    args = ap.parse_args()
    espresso_path = Path(SCRIPT_DIR / "espresso")

    espresso_out = Path(SCRIPT_DIR / "temp.out")
    
    run_espresso(args.input, espresso_out, espresso_path)
    pla = parse_pla(espresso_out)
    # exprs = pla_to_expressions(pla, not_style=args.not_style)

    print(f"=== Espresso output saved to: {espresso_out} ===\n")


    verilog_list = cube_to_gate(pla)
    print("******** FINAL RESULT ************")
    for code in verilog_list:
        print(code)
    
    output_file =SCRIPT_DIR / f"{args.module}.v"
    with open(output_file, "w") as f:
        for code in verilog_list:
            f.write(code + "\n")
    print(f"Module saved to {output_file}")
    
    espresso_out.unlink()
    print(f"=== Espresso output {espresso_out} is removed ===\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
