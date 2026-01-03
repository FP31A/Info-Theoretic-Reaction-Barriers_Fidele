import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, color='#EAECEE', edge='#2C3E50'):
    # Draw rectangle
    rect = patches.FancyBboxPatch((x, y), width, height, 
                                  boxstyle="round,pad=0.1", 
                                  ec=edge, fc=color, 
                                  linewidth=2)
    ax.add_patch(rect)
    # Add text
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=11, fontweight='bold', color='#2C3E50')

def draw_arrow(ax, x_start, y_start, x_end, y_end):
    ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", lw=2, color='#34495E'))

def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off') # Hide axes

    # --- 1. INPUT PHASE ---
    draw_box(ax, 0.5, 4.5, 2.5, 1, "Input Data\n(Transition1x)\n10k Reactions", color='#D6EAF8') # Blue tint
    
    # --- 2. PROCESSING PHASE ---
    draw_arrow(ax, 3.1, 5.0, 3.9, 5.0)
    draw_box(ax, 4.0, 4.5, 3.0, 1, "Geometry Parsing\n(HDF5 -> 3D Coords)", color='#EBDEF0') # Purple tint
    
    draw_arrow(ax, 5.5, 4.5, 5.5, 3.8)
    draw_box(ax, 4.0, 2.8, 3.0, 1, "High-Throughput QM\n(GFN2-xTB)\nRobust Wrapper", color='#EBDEF0')

    draw_arrow(ax, 5.5, 2.8, 5.5, 2.1)
    draw_box(ax, 4.0, 1.1, 3.0, 1, "Feature Engineering\n(Physics + Topology)", color='#EBDEF0')

    # --- 3. ANALYSIS PHASE (The Core) ---
    draw_arrow(ax, 7.1, 1.6, 7.9, 1.6)
    draw_box(ax, 8.0, 1.1, 3.5, 1, "Information-Theoretic\nFiltering\n(Mutual Information)", color='#F9E79F') # Yellow/Gold

    draw_arrow(ax, 9.75, 2.2, 9.75, 2.8)
    draw_box(ax, 8.0, 2.8, 3.5, 1, "Model Training\n(Random Forest)\nTop 10 Features", color='#F9E79F')

    # --- 4. OUTPUT PHASE ---
    draw_arrow(ax, 9.75, 3.9, 9.75, 4.5)
    draw_box(ax, 8.5, 4.5, 2.5, 1, "The Quantum Sieve\n(Screening/Ranking)", color='#A9DFBF', edge='#196F3D') # Green

    # --- LABELS ---
    plt.text(1.75, 5.7, "1. Data Curation", ha='center', fontsize=12, fontweight='bold', color='#7F8C8D')
    plt.text(5.5, 5.7, "2. Physics Engine (xTB)", ha='center', fontsize=12, fontweight='bold', color='#7F8C8D')
    plt.text(9.75, 5.7, "3. Analysis & Output", ha='center', fontsize=12, fontweight='bold', color='#7F8C8D')

    # Separation Lines
    ax.plot([3.5, 3.5], [0.5, 5.8], color='#BDC3C7', linestyle='--', alpha=0.5)
    ax.plot([7.5, 7.5], [0.5, 5.8], color='#BDC3C7', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("data/processed/ssef_visuals/Fig1_Pipeline_Architecture.png", dpi=300, bbox_inches='tight')
    print("Saved Fig1_Pipeline_Architecture.png")

if __name__ == "__main__":
    main()