"""
png2su.py - Modo terminal (mesmo uso do original).

USO:  python3 png2su.py imagem.png
Para a interface gráfica:  python3 png2su_gui.py
"""
import os
import sys

import png2su_core as core


def ask_vel(msg):
    while True:
        try:
            v = float(input(msg).replace(",", "."))
            if v > 0:
                return v
        except ValueError:
            pass
        print("  Digite um número positivo.")


def main():
    if len(sys.argv) != 2:
        sys.exit("USAGE: png2su.py image.png")
    png = sys.argv[1]
    if not os.path.isfile(png):
        sys.exit(f"Could not read {png}. USAGE: png2su.py image.png")

    print("##############################")
    print("######### PNG 2 SU #########")
    print("##############################\n")

    try:
        rgb = core.load_image(png)
    except ValueError as e:
        sys.exit(str(e))
    print("Image dimensions (Z, Y, RGB): ", rgb.shape)
    # Sem extensão das bordas: a simulação (simula_devito.py / interface) usa PML.

    colors, _ = core.detect_palette(rgb)
    print("\nNumber of velocities to be defined: ", len(colors))
    vels = []
    for R, G, B in colors:
        vels.append(ask_vel(
            f"Define the velocity related to the color "
            f"\x1b[38;2;{R};{G};{B}m(██████)\x1b[0m in km/s: "
        ))

    a, b, alpha = core.classify(rgb, colors)
    model = core.effective_velocity(a, b, alpha, vels)
    out = core.default_output(png)
    print("\nGenerating SU file...")
    header = core.write_su(model, out)
    print("SU file headers:\n")
    print(header)


if __name__ == "__main__":
    main()