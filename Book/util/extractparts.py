import sys
import subprocess
import xml.etree.ElementTree as ET

def extract_pieces(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}

    for group in root.findall(".//svg:g", namespaces):
        group_id = group.attrib.get('id', '')
        piece_name = group_id.removeprefix('piece-')
        if piece_name != group_id:
            filename = f"{group_id}.tex"

            # Extract bounding box for the piece
            piece_bbox = subprocess.run(
                ["inkscape", "--query-id", group_id, "--query-x", "--query-y", "--query-width", "--query-height", svg_file],
                capture_output=True, text=True).stdout.split()
            piece_x, piece_y = map(float, piece_bbox[:2])

            # Extract bounding box for the peg
            peg_id = f"ttpeg{piece_name}"
            peg_bbox = subprocess.run(
                ["inkscape", "--query-id", peg_id, "--query-x", "--query-y", "--query-width", "--query-height", svg_file],
                capture_output=True, text=True).stdout.split()
            peg_x, peg_y = map(float, peg_bbox[:2])
            print(piece_name, "Piece=", (piece_x, piece_y), " peg=", (peg_x, peg_y), file=sys.stderr)

            # Calculate offset
            offset_x = peg_x - piece_x
            offset_y = peg_y - piece_y

            # Generate PDF for the piece
            subprocess.run(["inkscape", f"--export-id-only", f"--export-id={group_id}", "--export-type=pdf", f"--export-filename={group_id}.pdf", svg_file])

            # Generate TeX output
            latex_output = (f"\\def\\TTGfx{piece_name.title()}" +
                            "#1{\\node[anchor=north west] at ($#1 + " +
                            f"({offset_x}, {offset_y})$) " +
                            f"{{\\includegraphics[width=\\TTXUnit, height=\\TTYUnit]" +
                            f"{{util/{group_id}.pdf}}}} }}")

            # Write latex output to file
            print(latex_output)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <svg_file>")
    else:
        extract_pieces(sys.argv[1])
