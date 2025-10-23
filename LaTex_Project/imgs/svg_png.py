import os
import sys
import argparse
from cairosvg import svg2png

def find_svgs(directory, recursive=False):
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(".svg"):
                yield os.path.join(root, name)
        if not recursive:
            break

def convert_svg_to_png(svg_path, overwrite=False):
    png_path = os.path.splitext(svg_path)[0] + ".png"
    if os.path.exists(png_path) and not overwrite:
        print(f"Skipping (exists): {png_path}")
        return
    try:
        svg2png(url=svg_path, write_to=png_path)
        print(f"Converted: {svg_path} -> {png_path}")
    except Exception as e:
        print(f"Error converting {svg_path}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Convert all SVG files in a folder to PNG.")
    parser.add_argument("dir", nargs="?", default=".", help="Directory to search (default: current dir)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing PNGs")
    args = parser.parse_args()

    target_dir = "./LaTex_Project/imgs"
    if not os.path.isdir(target_dir):
        print(f"Not a directory: {target_dir}", file=sys.stderr)
        sys.exit(1)

    svgs = list(find_svgs(target_dir, recursive=args.recursive))
    if not svgs:
        print("No SVG files found.")
        return

    for svg in svgs:
        convert_svg_to_png(svg, overwrite=args.force)

if __name__ == "__main__":
    main()