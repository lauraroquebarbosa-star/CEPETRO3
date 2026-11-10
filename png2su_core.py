"""
png2su_core.py - Núcleo do PNG 2 SU.

Converte uma imagem PNG (cada cor = uma camada) em um modelo de velocidades
no formato Seismic Unix (.su). Sem input() nem print(): pode ser usado tanto
pela interface gráfica quanto pelo modo terminal.
"""
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image


def load_image(path):
    """Abre o PNG e devolve um array (Z, Y, 3) uint8.

    Aceita RGBA/paleta desde que não haja pixels transparentes de verdade
    (o original recusava qualquer imagem com 4 canais).
    """
    img = Image.open(path)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        alpha = np.asarray(img.convert("RGBA"))[..., 3]
        if (alpha != 255).any():
            raise ValueError(
                "A imagem tem pixels transparentes. Exporte o PNG sem "
                "transparência (fundo opaco) e tente de novo."
            )
    return np.asarray(img.convert("RGB"))


def extend(rgb, extend_z, extend_y, mode="edge"):
    """Estende o modelo em Z e Y repetindo as bordas (equivale ao np.pad original)."""
    if extend_z == 0 and extend_y == 0:
        return rgb
    return np.pad(
        rgb,
        pad_width=((extend_z, extend_z), (extend_y, extend_y), (0, 0)),
        mode=mode,
    )


def find_colors(rgb):
    """Encontra as cores únicas.

    Em vez de converter cada pixel em string (lento), cada cor vira um único
    inteiro R*65536 + G*256 + B. O np.unique com return_inverse já devolve,
    para cada pixel, o índice da sua cor: o mapeamento cor -> velocidade
    depois é só uma indexação de array.

    Retorna:
        colors    (n, 3) uint8 com as cores únicas
        index_map (Z, Y) com o índice da cor de cada pixel
        counts    (n,) número de pixels de cada cor
    """
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    keys = (r << 16) | (g << 8) | b

    uniq, inverse, counts = np.unique(
        keys.ravel(), return_inverse=True, return_counts=True
    )
    colors = np.stack(
        [(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1
    ).astype(np.uint8)
    return colors, inverse.reshape(keys.shape), counts


def velocity_model(index_map, vels):
    """Troca o índice de cor de cada pixel pela velocidade (km/s)."""
    return np.asarray(vels, dtype=np.float32)[index_map]


def _run(cmd):
    """Roda um comando de shell e levanta erro legível se falhar."""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Falha no comando:\n{cmd}\n\n{res.stderr.strip()}")
    return res.stdout


def write_su(model, output_path):
    """Gera o arquivo .su a partir do modelo (Z, Y) de velocidades.

    O original escrevia um .txt e chamava o a2b para virar binário. Aqui os
    floats são gravados direto em binário (mesmo resultado do a2b), o que
    elimina o arquivo de texto e o tempo de leitura/escrita dele.
    Arquivos temporários ficam numa pasta temporária, não na pasta do usuário.

    Retorna o cabeçalho do surange (texto) para mostrar ao usuário.
    """
    nz, ny = model.shape
    output_path = os.path.abspath(output_path)

    with tempfile.TemporaryDirectory() as tmp:
        dat = os.path.join(tmp, "png2su_vels.dat")
        tmp_su = os.path.join(tmp, "temp.su")

        # Transposta: um traço por posição Y, cada um com nz amostras (como no original)
        np.ascontiguousarray(model.T, dtype=np.float32).tofile(dat)

        # TODO: a linha 127 original continua depois de "suchw key1=scalco ..."
        # e ficou cortada na foto. Cole aqui o restante dela (entre selev e >temp.su).
        _run(
            f'suaddhead <"{dat}" ns={nz} '
            f"| suchw key1=dt a=1 b=0 c=0 "
            f"| suchw key1=selev a={nz} "
            f'>"{tmp_su}"'
        )
        _run(
            f'suchw <"{tmp_su}" key1=sy key2=tracl a=-1 b=1 c=0 '
            f"| suchw key1=trid a=0 "
            f"| suchw key1=tracl a=0 "
            f'>"{output_path}"'
        )

    header = _run(f'surange <"{output_path}" | head -n 7')
    return header


def default_output(png_path):
    """imagem.png -> imagem.su (mesma pasta), como no original."""
    base, _ = os.path.splitext(png_path)
    return base + ".su"


# =====================================================================
# v2 - Funções da análise matemática (ver documento "PNG → SU: análise")
# =====================================================================

def detect_palette(rgb, min_frac=0.002, merge_dist=12.0):
    """Cores 'de verdade' das camadas: as que ocupam pelo menos min_frac dos pixels.

    Num PNG com antialiasing, as bordas criam centenas de cores de mistura,
    cada uma com poucos pixels. As camadas reais são poucas e grandes.
    """
    colors, _, counts = find_colors(rgb)
    keep = counts >= min_frac * counts.sum()
    cand, cc = colors[keep].astype(np.float64), counts[keep]
    # funde cores quase iguais (distância RGB < merge_dist), mantendo a mais frequente
    pal, pcount = [], []
    for i in np.argsort(-cc):
        for j, p in enumerate(pal):
            if np.sqrt(((cand[i] - p) ** 2).sum()) < merge_dist:
                pcount[j] += cc[i]
                break
        else:
            pal.append(cand[i]); pcount.append(cc[i])
    return np.array(pal).round().astype(np.uint8), np.array(pcount)


def classify(rgb, palette):
    """Para cada pixel: as 2 cores da paleta mais próximas (a, b) e a fração alpha de b.

    Um pixel de borda com antialiasing é (aprox.) a mistura linear
        c = (1 - alpha) * P_a + alpha * P_b
    alpha = projeção de (c - P_a) no segmento P_a -> P_b, limitada a [0, 1].
    Pixels "puros" saem com alpha = 0.
    """
    P = np.asarray(palette, dtype=np.float32)
    c_all = rgb.reshape(-1, 3).astype(np.float32)
    n = c_all.shape[0]
    a = np.empty(n, np.int32); b = np.empty(n, np.int32); alpha = np.empty(n, np.float32)
    P2 = (P ** 2).sum(-1)
    for i in range(0, n, 500_000):                 # em blocos: memória O(bloco x ncores)
        c = c_all[i:i + 500_000]
        d = (c ** 2).sum(-1)[:, None] - 2 * c @ P.T + P2[None, :]
        o = np.argsort(d, axis=1)
        ia = o[:, 0]; ib = o[:, 1] if P.shape[0] > 1 else o[:, 0]
        seg = P[ib] - P[ia]
        L2 = (seg ** 2).sum(-1)
        al = ((c - P[ia]) * seg).sum(-1) / np.maximum(L2, 1e-12)
        a[i:i + 500_000], b[i:i + 500_000] = ia, ib
        alpha[i:i + 500_000] = np.where(L2 > 0, np.clip(al, 0, 1), 0)
    shp = rgb.shape[:2]
    return a.reshape(shp), b.reshape(shp), alpha.reshape(shp)


def effective_velocity(a, b, alpha, vels, rho=None):
    """Velocidade efetiva por pixel: média HARMÔNICA do módulo K = rho v^2.

    1/K_ef = (1-alpha)/K_a + alpha/K_b   (meio efetivo de Backus / Moczo et al.)
    Com densidade constante: v_ef = 1 / sqrt((1-alpha)/v_a^2 + alpha/v_b^2).
    """
    v = np.asarray(vels, dtype=np.float64)
    va, vb = v[a], v[b]
    return (1.0 / np.sqrt((1 - alpha) / va ** 2 + alpha / vb ** 2)).astype(np.float32)


def resample_harmonic(vel, factor):
    """Reduz a resolução por um fator inteiro com média harmônica de K (não 'vizinho mais próximo').

    Uma célula da malha que cobre factor x factor pixels recebe
        v_ef = 1 / sqrt( mean(1 / v^2) ).
    """
    f = int(factor)
    nz, ny = (vel.shape[0] // f) * f, (vel.shape[1] // f) * f
    s = 1.0 / vel[:nz, :ny].astype(np.float64) ** 2
    s = s.reshape(nz // f, f, ny // f, f).mean(axis=(1, 3))
    return (1.0 / np.sqrt(s)).astype(np.float32)


def grid_requirements(vmin, vmax, f_peak, ppw=8.0, courant=0.5):
    """dx e dt máximos para a simulação.

    Dispersão: dx <= vmin / (ppw * f_max), com f_max ~ 2.5 f_peak (Ricker).
    Estabilidade (CFL): dt <= courant * dx / vmax.
    ppw = 8 é típico para FD de 4ª ordem; ~10-12 para 2ª ordem.
    courant = 0.5 é seguro para FD 4ª ordem 2D (limite ~0.606).
    """
    fmax = 2.5 * f_peak
    dx = vmin / (ppw * fmax)
    dt = courant * dx / vmax
    return dict(f_max=fmax, dx_max=dx, dt_max=dt)


def min_padding(vmax, t_record, d_shallow, dx):
    """Nº mínimo de células de extensão SEM camada absorvente.

    A reflexão na borda percorre pelo menos 2 (d + w): da fonte até a borda e
    de volta ao receptor, com d = menor distância de fonte/receptor à borda original.
    Para ela chegar só depois do fim do registro:  2 (d + w) / vmax >= T
        =>  w >= vmax T / 2 - d
    """
    w = vmax * t_record / 2.0 - d_shallow
    return max(0, int(np.ceil(w / dx)))