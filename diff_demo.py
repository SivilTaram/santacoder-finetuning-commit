import difflib

def generate_line_diff(original_code, modified_code):
    original_lines = original_code.splitlines(keepends=True)
    modified_lines = modified_code.splitlines(keepends=True)

    # Add line numbers to original and modified code
    original_code_with_line_numbers = ''.join(f"{i + 1:3d} {line}" for i, line in enumerate(original_lines))
    modified_code_with_line_numbers = ''.join(f"{i + 1:3d} {line}" for i, line in enumerate(modified_lines))

    diff = difflib.unified_diff(original_code_with_line_numbers.splitlines(),
                                modified_code_with_line_numbers.splitlines(), 
                                fromfile='Original', 
                                tofile='Modified', 
                                lineterm='')
    # remove the first line and second line
    # print(diff)
    try:
        next(diff)
        next(diff)
        diff_with_line_numbers = '\n'.join(f"{line}" for line in diff if line.startswith(('+', '-')))
        return diff_with_line_numbers
    except StopIteration:
        return ""


def apply_line_diff(original_code, line_diff):
    original_lines = original_code.splitlines(keepends=True)
    modified_lines = original_lines

    diff_lines = line_diff.splitlines()
    # Iterate over the diff lines and apply the changes to reconstruct the modified code
    for line in diff_lines:
        line_idx = int(line[1:4].strip()) - 1
        if line.startswith('-'):
            # fetch idx
            modified_lines[line_idx] = ""
        elif line.startswith('+'):
            if line_idx > len(modified_lines) - 1:
                modified_lines.append(line[5:] + "\n")
            else:
                modified_lines[line_idx] = line[5:] + "\n"

    
    modified_code = ''.join(modified_lines)
    return modified_code

# Example usage
original_code = """
def multiply(a, b):
    return a * b
"""

modified_code = """
def multiply(a, b):
    return a * b   """

line_diff = generate_line_diff(original_code, modified_code)
new_code = apply_line_diff(original_code, line_diff)
# print(line_diff)
print(new_code)