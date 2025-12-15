import os
import networkx as nx
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont

EXCLUDE_DIRS = {'.vscode', '__pycache__'}
EXCLUDE_FILES = {'.DS_Store'}
EXCLUDE_EXTENSIONS = {'.pyc', '.pyo', '.log'}


def visualize_directory_as_graph(root_path='.'):
    """
    Visualizes the directory structure as a directed graph using NetworkX and Matplotlib.
    Args:
        root_path (str): The root directory to visualize. Defaults to the current directory.
    Returns:
        None: Displays a graph of the directory structure.
    """
    def build_graph(path, graph, parent=None):
        """
        Recursively builds a directed graph from the directory structure.
        Args:
            path (str): The current directory path.
            graph (nx.DiGraph): The directed graph to build.
            parent (str): The parent node in the graph.
        Returns:
            None: Modifies the graph in place.
        """
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            graph.add_node(full_path, label=entry)
            if parent:
                graph.add_edge(parent, full_path)
            if os.path.isdir(full_path):
                build_graph(full_path, graph, full_path)

    G = nx.DiGraph()
    build_graph(root_path, G)
    pos = nx.spring_layout(G, k=0.5, iterations=100)
    labels = {node: os.path.basename(node) for node in G.nodes()}
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=False, node_size=50)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title(f"Directory Tree Graph: {os.path.abspath(root_path)}")
    plt.axis('off')
    plt.show()



def should_exclude(name, full_path):
    if name in EXCLUDE_FILES:
        return True
    if name in EXCLUDE_DIRS and os.path.isdir(full_path):
        return True
    if any(name.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
        return True
    return False

def generate_ascii_tree(root_path='.', prefix=''):
    lines = []
    try:
        entries = sorted(os.listdir(root_path))
    except PermissionError:
        return [f"{prefix}|-- [Permission Denied]"]

    entries = [e for e in entries if not should_exclude(e, os.path.join(root_path, e))]

    for i, entry in enumerate(entries):
        path = os.path.join(root_path, entry)
        connector = "`-- " if i == len(entries) - 1 else "|-- "
        lines.append(f"{prefix}{connector}{entry}")
        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "|   "
            lines.extend(generate_ascii_tree(path, prefix + extension))
    return lines

def save_ascii_tree_image(output_path='ascii_tree.png', root_path='.', font_size=14):
    lines = generate_ascii_tree(root_path)
    if not lines:
        lines = ["[Empty directory]"]

    font = ImageFont.load_default()
    padding = 10
    line_height = font_size + 4
    width = max(font.getlength(line) for line in lines) + 2 * padding
    height = line_height * len(lines) + 2 * padding

    image = Image.new("RGB", (int(width), int(height)), color="white")
    draw = ImageDraw.Draw(image)

    for i, line in enumerate(lines):
        draw.text((padding, padding + i * line_height), line, font=font, fill="black")

    image.save(output_path)
    print(f"Saved clean tree image to: {output_path}")
