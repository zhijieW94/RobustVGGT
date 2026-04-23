#!/usr/bin/env python3
"""Convert image_list.txt files from numbered format to standard format.

Before: 1	[Clean] images/DSC_0286.JPG
After:  [Clean] images/DSC_0286.JPG
"""

import sys
from pathlib import Path


def convert_image_list(txt_path: Path) -> tuple[int, str]:
    """Convert a single image_list.txt file. Returns (n_lines_converted, status)."""
    if not txt_path.is_file():
        return 0, "file not found"

    lines = txt_path.read_text().splitlines()
    converted = []
    had_numbers = False

    for line in lines:
        line = line.strip()
        if not line:
            converted.append("")
            continue

        # Check if line starts with a number followed by whitespace and then [Clean] or [Noisy]
        tokens = line.split(None, 1)  # Split on first whitespace
        if len(tokens) >= 2 and tokens[0].isdigit() and ("[Clean]" in tokens[1] or "[Noisy]" in tokens[1]):
            # Remove the number prefix
            converted.append(tokens[1])
            had_numbers = True
        else:
            converted.append(line)

    if not had_numbers:
        return 0, "already in correct format"

    # Write back
    txt_path.write_text("\n".join(converted) + "\n")
    return len([l for l in converted if "[Clean]" in l or "[Noisy]" in l]), "converted"


def main():
    dataset_root = Path("/nvmepool/zhijiewu/Datasets/Final_Benchmarks")

    if not dataset_root.is_dir():
        print(f"Error: dataset root not found: {dataset_root}")
        sys.exit(1)

    total_converted = 0
    total_files = 0

    # Process all image_list.txt files in clean datasets
    for txt_file in sorted(dataset_root.glob("clean/*/*/image_list.txt")):
        total_files += 1
        n_converted, status = convert_image_list(txt_file)
        total_converted += n_converted
        rel_path = txt_file.relative_to(dataset_root)
        print(f"  {rel_path}: {status} ({n_converted} lines)")

    print(f"\nTotal: {total_files} files processed, {total_converted} lines converted")


if __name__ == "__main__":
    main()
