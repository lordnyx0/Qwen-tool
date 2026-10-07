import sys
sys.path.insert(0, 'scripts')
from benchmark_90_battery_q8 import extract_tool_calls_from_text, test_python_code

def sanitize_generated_code(code: str) -> str:
    code = code.strip()
    while code and code[-1] in ['"', "'", '.', '}', ']', ';']:
        code = code[:-1].strip()
    open_paren = code.count('(')
    close_paren = code.count(')')
    while close_paren > open_paren and code.endswith(')'):
        code = code[:-1].strip()
        close_paren -= 1
    return code

sample = '''<think>
O usuário solicitado realizará a realizar a operação exatamente 8^145 mod 17 utilizando a operação exatamente.
</think>

<tool_call>
{"name": "execute_python", "arguments": {"code": "print(8**145 % 17))"."}}
</tool_call>'''

calls, has_xml, errs = extract_tool_calls_from_text(sample)
code = calls[0]["arguments"]["code"]
clean_code = sanitize_generated_code(code)
print("Original code:", repr(code))
print("Cleaned code: ", repr(clean_code))

ok, msg = test_python_code(clean_code, 8)
print("Execution:", ok, msg)
