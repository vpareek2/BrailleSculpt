import subprocess
from os import listdir, remove
from os.path import expanduser, join

# Define a function to convert Braille text to a 3D model
def braille_to_scad(filename, braille_text):
    # Define the standard dimensions for Braille dots
    dot_diameter = 1.5  # mm
    dot_spacing = 2.5  # mm between dots in a cell

    with open(f"{filename}.scad", "w") as file:
        # Write OpenSCAD code to create a 3D model
        file.write("module BrailleDot() {\n")
        file.write(f"    cylinder(h=1, d={dot_diameter}, $fn=20);\n")
        file.write("}\n\n")

        # Process each character in the Braille text
        for line_index, line in enumerate(braille_text.splitlines()):
            for char_index, char in enumerate(line):
                if char != " ":
                    # Convert the Braille character to dot positions
                    dot_positions = braille_char_to_dots(char)
                    for x, y in dot_positions:
                        # Calculate the position of each dot
                        dot_x = char_index * 6 * dot_spacing + x * dot_spacing
                        dot_y = line_index * 4 * dot_spacing + y * dot_spacing
                        file.write(f"translate([{dot_x}, {dot_y}, 0]) BrailleDot();\n")

def braille_char_to_dots(char):
    braille_dict = {
        '⠁': [(0, 0)],
        '⠃': [(0, 0), (0, 1)],
        '⠉': [(0, 0), (1, 0)],
        '⠙': [(0, 0), (1, 0), (1, 1)],
        '⠑': [(0, 0), (1, 1)],
        '⠋': [(0, 0), (0, 1), (1, 0)],
        '⠛': [(0, 0), (0, 1), (1, 0), (1, 1)],
        '⠓': [(0, 0), (0, 1), (1, 1)],
        '⠊': [(0, 1), (1, 0)],
        '⠚': [(0, 1), (1, 0), (1, 1)],
        '⠅': [(0, 0), (1, 2)],
        '⠇': [(0, 0), (0, 1), (1, 2)],
        '⠍': [(0, 0), (1, 0), (1, 2)],
        '⠝': [(0, 0), (1, 0), (1, 1), (1, 2)],
        '⠕': [(0, 0), (1, 1), (1, 2)],
        '⠏': [(0, 0), (0, 1), (1, 0), (1, 2)],
        '⠟': [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)],
        '⠗': [(0, 0), (0, 1), (1, 1), (1, 2)],
        '⠎': [(0, 1), (1, 0), (1, 2)],
        '⠞': [(0, 1), (1, 0), (1, 1), (1, 2)],
        '⠥': [(0, 0), (1, 2), (2, 0)],
        '⠧': [(0, 0), (0, 1), (1, 2), (2, 0)],
        '⠺': [(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)],
        '⠭': [(0, 0), (1, 0), (1, 2), (2, 0)],
        '⠽': [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)],
        '⠵': [(0, 0), (1, 1), (1, 2), (2, 0)],
        ' ': []
    }
    # Return the dot positions for the given character
    # If the character is not in the dictionary, return an empty list
    return braille_dict.get(char, [])


def open_in_stl_viewer(file_path):
    # Replace 'STLViewerApp' with the actual name or path of your STL Viewer application
    stl_viewer_app = 'STL Viewer.app'
    subprocess.run(['open', '-a', stl_viewer_app, file_path])

def scad_to_stl(filename):
    files = listdir('.')
    downloads_folder = expanduser('~/Downloads')  # Path to the Downloads folder

    for f in files:
        if f == f"{filename}.scad":
            stl_filename = f.replace('.scad', '.stl')
            of = join(downloads_folder, stl_filename)  # Path to the STL file in Downloads folder
            # Specify the full path to the OpenSCAD executable
            cmd = ["/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD", "--export-format=binstl", "--render", "-o", of, f]
            subprocess.run(cmd)

            # Open the STL file in STL Viewer
            open_in_stl_viewer(of)

    remove(f"{filename}.scad")



# Example usage
# braille_text = "⠓⠑⠇⠇⠕ ⠊ ⠁⠍ ⠞⠓⠑ ⠇⠕⠋⠊ ⠛⠊⠗"
# braille_to_scad("braille_output", braille_text)
# scad_to_stl("braille_output")