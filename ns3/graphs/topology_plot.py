import matplotlib.pyplot as plt
import re

def read_coordinates(filename):
    gnbs = []
    ues = []
    pattern = re.compile(r'id="(\d+)"[^>]*locX="([\d\.]+)"[^>]*locY="([\d\.]+)"')

    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                node_id = int(match.group(1))
                x = float(match.group(2))
                y = float(match.group(3))
                if node_id < 12:
                    gnbs.append((node_id, x, y))
                else:
                    ues.append((node_id, x, y))
    return gnbs, ues

def plot_nodes(gnbs, ues):
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot gNBs
    for node_id, x, y in gnbs:
        ax.scatter(x, y, c='red', marker='^', s=100, label='gNB' if node_id == 0 else "")
        ax.text(x + 1, y + 1, f"gNB {node_id}", fontsize=8, color='red')

    # Plot UEs
    for node_id, x, y in ues:
        ax.scatter(x, y, c='blue', marker='o', s=50, label='UE' if node_id == 12 else "")
        if node_id % 3 == 0 or node_id == 111:  # only label some to avoid clutter
            ax.text(x + 1, y + 1, f"UE {node_id}", fontsize=7, color='blue')

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("gNB and UE Node Positions")
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('network_layout.png')
    plt.show()

if __name__ == "__main__":
    filename = "topology_test.xml"
    gnbs, ues = read_coordinates(filename)
    plot_nodes(gnbs, ues)
