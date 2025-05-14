import shutil
import pyfiglet

def align_text(text, alignment, width):
    """
    Align each line of a multi-line string.

    :param text: The multi-line string to align.
    :param alignment: "center", "right", or "left".
    :param width: The width for alignment.
    :return: The aligned multi-line string.
    """
    lines = text.splitlines()
    if alignment == "center":
        return "\n".join([line.center(width) for line in lines])
    elif alignment == "right":
        return "\n".join([line.rjust(width) for line in lines])
    elif alignment == "left":
        return "\n".join([line.ljust(width) for line in lines])
    else:
        return text


def print_banner():
    # Get terminal width (default to 80 if undetermined)
    term_width = shutil.get_terminal_size((80, 20)).columns

    # Create a border line using '=' characters
    border = "=" * term_width

    # Generate ASCII art for the title using the 'alpha' font.
    title = pyfiglet.figlet_format("Transfer-PIDL", font="slant")

    # Define additional banner texts
    sub_title = "Physics-informed Deep Learning with Transfer Learning"
    author = "Developed by Muyuan Liu A.K.A. Louis L."
    address = "at Imperial College London"

    # Generate a logo using pyfiglet with the 'slant' font.
    logo = pyfiglet.figlet_format("I", font="5lineoblique")

    # Print the banner with borders and aligned text.
    print(border)
    # Center the title (which may be multi-line)
    print(align_text(title, "center", term_width))
    print('')
    # Right-align the subtitle, author, and address texts
    print(sub_title.center(term_width))
    print(author.center(term_width))
    print(address.center(term_width))
    print('')
    # Right-align the logo (which is multi-line)
    print(align_text(logo, "center", term_width))
    print(border)