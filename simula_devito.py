#!/usr/bin/env python3
"""
simula_devito.py - Simulação acústica 2D com Devito a partir do modelo do png2su.

Lê o modelo de velocidades (.su gerado pelo png2su, ou .npy), simula um ou
vários tiros com borda absorvente e salva um sismograma .su por tiro.

EXEMPLO
  python3 simula_devito.py modelo.su --dx 5 --f0 20 --tmax 1.0 \
      --sx 100:900:100 --sz 10 --rx 0:1000:5 --rz 10 --procs 4 --out tiros

  --sx ini:fim:passo   posições x das fontes em metros (aqui: 100, 200, ..., 900)
  --rx ini:fim:passo   posições x dos receptores em metros
  --sz / --rz          profundidade das fontes / receptores em metros

O modelo NÃO precisa ser estendido no png2su (deixe Z e Y = 0): a camada
absorvente PML é adicionada aqui, fora do modelo.

Unidades: velocidades do png2su em km/s; aqui tudo vira metros e segundos.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np


# ------------------------------------------------------------------ modelo
def read_su(path):
    """Lê um .su (cabeçalho de 240 bytes + ns floats por traço). Retorna (ntraços, ns)."""
    raw = np.fromfile(path, dtype=np.uint8)
    ns = int(np.frombuffer(raw[114:116].tobytes(), dtype=np.uint16)[0])
    tr = 240 + 4 * ns
    if ns == 0 or raw.size % tr:
        raise ValueError(f"{path}: não parece um arquivo SU válido (ns={ns}).")
    data = raw.reshape(-1, tr)[:, 240:].copy().view(np.float32)
    return data                                    # (Y, Z): um traço por posição Y


def load_model(path):
    """Retorna velocidade em m/s com forma (nx, nz) = (Y do png2su, Z)."""
    if path.endswith(".npy"):
        v = np.load(path).T                        # png2su: (Z, Y) -> (nx, nz)
    else:
        v = read_su(path)
    v = v.astype(np.float32)
    if v.max() < 20:                               # km/s -> m/s
        v = v * 1000.0
    return v


# ------------------------------------------------------------------ escrita SU
def write_su(path, traces, dt, fldr, sx, gx, sz, rz):
    """Grava sismograma (nrec, nt) em SU nativo (little-endian), um traço por receptor."""
    nrec, nt = traces.shape
    hdr = np.zeros((nrec, 60), dtype=np.int32)     # 240 bytes = 60 int32
    h16 = hdr.view(np.int16)                       # (nrec, 120)
    hdr[:, 0] = np.arange(1, nrec + 1)             # tracl
    hdr[:, 1] = np.arange(1, nrec + 1)             # tracr
    hdr[:, 2] = fldr                               # fldr (nº do tiro)
    hdr[:, 3] = np.arange(1, nrec + 1)             # tracf
    h16[:, 14] = 1                                 # trid = 1 (sísmico)
    hdr[:, 9] = np.round(gx - sx).astype(np.int32)  # offset
    hdr[:, 10] = -int(round(rz))                   # gelev
    hdr[:, 11] = -int(round(sz))                   # selev
    hdr[:, 12] = int(round(sz))                    # sdepth
    h16[:, 34] = 1                                 # scalel
    h16[:, 35] = 1                                 # scalco
    hdr[:, 18] = int(round(sx))                    # sx
    hdr[:, 20] = np.round(gx).astype(np.int32)     # gx
    h16[:, 57] = nt                                # ns (byte 114)
    h16[:, 58] = int(round(dt * 1e6))              # dt em microssegundos (byte 116)
    out = np.concatenate([hdr.view(np.uint8), np.ascontiguousarray(traces, dtype=np.float32).view(np.uint8)], axis=1)
    out.tofile(path)


# ------------------------------------------------------------------ numérica
def fd1_stag_weights(order):
    """Pesos da 1ª derivada em malha intercalada (pontos ±1/2, ±3/2, ...), ordem 'order' (Fornberg)."""
    m = order // 2
    x = np.arange(-m, m) + 0.5
    A = np.vander(x, increasing=True).T
    b = np.zeros(len(x)); b[1] = 1.0
    return np.linalg.solve(A, b)


def stable_dt(dx, vmax, order, safety=0.8):
    """Maior dt estável (velocidade-pressão intercalado, 2D): dt <= dx / (vmax sqrt(2) sum|c_k|)."""
    w = fd1_stag_weights(order)
    return safety * dx / (vmax * np.sqrt(2.0) * np.abs(w).sum())


def ricker(f0, t):
    a = (np.pi * f0 * (t - 1.5 / f0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def pml_profile(n, nbl, dx, vmax, R=1e-5, skip_start=False):
    """Amortecimento da PML [1/s] ao longo de um eixo: d = d0 (dist/L)^2, d0 = 3 vmax ln(1/R) / (2L)."""
    d = np.zeros(n)
    if nbl == 0:
        return d
    L = nbl * dx
    d0 = 3.0 * vmax * np.log(1.0 / R) / (2.0 * L)
    i = np.arange(nbl)
    d[:nbl] = d0 * ((nbl - i) / nbl) ** 2
    d[n - nbl:] = d[:nbl][::-1]
    if skip_start:
        d[:nbl] = 0.0
    return d


# ------------------------------------------------------------------ um tiro
_G = {}


def _setup(cfg):
    """Monta (uma vez por processo) malha, campos e operador do Devito.

    Formulação de 1ª ordem velocidade-pressão em malha intercalada, com PML de
    campo dividido (Berenger): p = px + pz, cada parte amortecida só na sua direção.
        dvx/dt + dx(x) vx = -(1/rho) dp/dx        dpx/dt + dx(x) px = -K dvx/dx
        dvz/dt + dz(z) vz = -(1/rho) dp/dz        dpz/dt + dz(z) pz = -K dvz/dz
    Densidade constante (rho = 1), K = v^2.
    """
    from devito import (Grid, Function, TimeFunction, VectorTimeFunction, SparseTimeFunction,
                        Eq, Operator, NODE)

    v, dx, nbl, so = cfg["v"], cfg["dx"], cfg["nbl"], cfg["so"]
    vp = np.pad(v, nbl, mode="edge")
    nx, nz = vp.shape
    grid = Grid(shape=(nx, nz), extent=((nx - 1) * dx, (nz - 1) * dx),
                origin=(-nbl * dx, -nbl * dx), dtype=np.float32)
    x, z = grid.dimensions
    dt = cfg["dt"]

    K = Function(name="K", grid=grid, space_order=so); K.data[:] = vp.astype(np.float32) ** 2
    vmax = float(v.max())
    dxp = pml_profile(nx, nbl, dx, vmax)
    dzp = pml_profile(nz, nbl, dx, vmax, skip_start=cfg["free_surface"])
    # fatores de atualização semi-implícitos: a = (1 - d dt/2)/(1 + d dt/2), b = dt/(1 + d dt/2)
    ax = Function(name="ax", grid=grid); bx = Function(name="bx", grid=grid)
    az = Function(name="az", grid=grid); bz = Function(name="bz", grid=grid)
    ax.data[:] = ((1 - 0.5 * dt * dxp) / (1 + 0.5 * dt * dxp))[:, None]
    bx.data[:] = (dt / (1 + 0.5 * dt * dxp))[:, None]
    az.data[:] = ((1 - 0.5 * dt * dzp) / (1 + 0.5 * dt * dzp))[None, :]
    bz.data[:] = (dt / (1 + 0.5 * dt * dzp))[None, :]

    px = TimeFunction(name="px", grid=grid, staggered=NODE, space_order=so, time_order=1)
    pz = TimeFunction(name="pz", grid=grid, staggered=NODE, space_order=so, time_order=1)
    vel = VectorTimeFunction(name="v", grid=grid, space_order=so, time_order=1)
    vx, vz = vel[0], vel[1]
    p = px + pz

    eqs = [Eq(vx.forward, ax * vx - bx * p.dx),
           Eq(vz.forward, az * vz - bz * p.dy),
           Eq(px.forward, ax * px - bx * K * vx.forward.dx),
           Eq(pz.forward, az * pz - bz * K * vz.forward.dy)]
    if cfg["free_surface"]:                       # p = 0 na superfície (z = 0) e acima dela
        mask = Function(name="mask", grid=grid); mask.data[:] = 1.0
        mask.data[:, :nbl + 1] = 0.0
        eqs[2] = Eq(px.forward, mask * (ax * px - bx * K * vx.forward.dx))
        eqs[3] = Eq(pz.forward, mask * (az * pz - bz * K * vz.forward.dy))

    nt = cfg["nt"]
    src = SparseTimeFunction(name="src", grid=grid, npoint=1, nt=nt)
    rec = SparseTimeFunction(name="rec", grid=grid, npoint=len(cfg["rx"]), nt=nt)
    rec.coordinates.data[:, 0] = cfg["rx"]
    rec.coordinates.data[:, 1] = cfg["rz"]
    inj = (src.inject(field=px.forward, expr=0.5 * src * dt / (dx * dx))
           + src.inject(field=pz.forward, expr=0.5 * src * dt / (dx * dx)))
    rd = rec.interpolate(expr=px + pz)
    op = Operator(eqs + inj + rd, subs=grid.spacing_map)
    _G.update(grid=grid, fields=(px, pz, vx, vz), src=src, rec=rec, op=op)


def run_shot(args):
    i, sx, cfg = args
    if not _G:
        _setup(cfg)
    src, rec, op = _G["src"], _G["rec"], _G["op"]
    for f in _G["fields"]:
        f.data[:] = 0.0
    src.coordinates.data[0, :] = (sx, cfg["sz"])
    src.data[:, 0] = cfg["wavelet"]
    rec.data[:] = 0.0
    t = time.time()
    op.apply(dt=cfg["dt"], time_M=cfg["nt"] - 2)
    seis = rec.data[::cfg["k"], :].T.copy()       # (nrec, nt_out)
    path = os.path.join(cfg["out"], f"tiro_{i + 1:03d}_sx{int(round(sx))}.su")
    write_su(path, seis, cfg["dt"] * cfg["k"], i + 1, sx, cfg["rx"], cfg["sz"], cfg["rz"][0])
    return path, time.time() - t


# ------------------------------------------------------------------ planejamento e execução
def rng(s):
    """'100' -> [100];  '0:1000:5' -> [0, 5, ..., 1000]."""
    a = [float(x) for x in str(s).split(":")]
    if len(a) == 1:
        return np.array(a)
    if len(a) != 3 or a[2] <= 0:
        raise ValueError(f"Intervalo inválido: '{s}'. Use inicio:fim:passo.")
    return np.arange(a[0], a[1] + a[2] / 2, a[2])


def plan(v, dx, f0, tmax, sx, sz, rx, rz, dt_out=0.001, nbl=20, so=8,
         free_surface=False, out="tiros"):
    """Monta a configuração da simulação e confere tudo antes de rodar.

    v: velocidades em m/s, forma (nx, nz) = (Y, Z).
    Retorna (cfg, info). info["errors"] não vazio = não dá para rodar.
    """
    v = np.asarray(v, dtype=np.float32)
    nx, nz = v.shape
    sx, rx = np.atleast_1d(np.asarray(sx, float)), np.atleast_1d(np.asarray(rx, float))
    vmin, vmax = float(v.min()), float(v.max())
    info = dict(nx=nx, nz=nz, width=(nx - 1) * dx, depth=(nz - 1) * dx, vmin=vmin, vmax=vmax,
                nshots=len(sx), nrec=len(rx), errors=[], warnings=[])

    ppw = 8.0 if so <= 4 else 5.0
    info["dx_max"] = dx_max = vmin / (ppw * 2.5 * f0)
    if dx > dx_max:
        info["warnings"].append(
            f"Pixel de {dx:g} m é grande para {f0:g} Hz: o limite é {dx_max:.2f} m "
            f"(dispersão numérica). Use um PNG mais fino ou uma frequência menor.")
    dtc = stable_dt(dx, vmax, so)
    k = int(np.ceil(dt_out / dtc))
    dt = dt_out / k
    nt = int(round(tmax / dt)) + 1
    info.update(dt=dt, dt_stable=dtc, nt=nt, k=k, nt_out=(nt - 1) // k + 1)
    info["cells"] = (nx + 2 * nbl) * (nz + 2 * nbl)

    L, D = info["width"], info["depth"]
    if len(sx) == 0 or len(rx) == 0:
        info["errors"].append("Defina pelo menos uma fonte e um receptor.")
    else:
        if sx.min() < 0 or sx.max() > L:
            info["errors"].append(f"Fontes fora do modelo (x deve estar entre 0 e {L:g} m).")
        if rx.min() < 0 or rx.max() > L:
            info["errors"].append(f"Receptores fora do modelo (x deve estar entre 0 e {L:g} m).")
    if not (0 <= sz <= D) or not (0 <= rz <= D):
        info["errors"].append(f"Profundidade fora do modelo (z deve estar entre 0 e {D:g} m).")
    if free_surface and (sz < dx or rz < dx):
        info["warnings"].append("Com superfície livre, fonte ou receptor em z = 0 registram zero: "
                                "coloque-os pelo menos 1 pixel abaixo.")
    if info["nt_out"] > 32767:
        info["errors"].append("Registro longo demais para o formato SU (máx. 32767 amostras): "
                              "aumente a amostragem de saída ou reduza a duração.")

    t = np.arange(nt) * dt
    cfg = dict(v=v, dx=dx, nbl=nbl, so=so, nt=nt, dt=dt, k=k, sz=float(sz), sx=sx,
               rx=rx, rz=np.full(len(rx), float(rz)),
               wavelet=(np.cumsum(ricker(f0, t)) * dt).astype(np.float32),  # fonte de taxa: integra o Ricker
               out=out, free_surface=free_surface)
    return cfg, info


def run(cfg, procs=1, callback=None):
    """Roda todos os tiros. callback(feitos, total, caminho, segundos) após cada um."""
    import multiprocessing as mp
    os.makedirs(cfg["out"], exist_ok=True)
    jobs = [(i, float(s), cfg) for i, s in enumerate(cfg["sx"])]
    procs = max(1, min(procs, len(jobs)))
    res = []
    if procs == 1:
        _G.clear()
        for j in jobs:
            r = run_shot(j)
            res.append(r)
            if callback:
                callback(len(res), len(jobs), *r)
    else:
        with mp.get_context("spawn").Pool(procs) as pool:
            for r in pool.imap(run_shot, jobs):
                res.append(r)
                if callback:
                    callback(len(res), len(jobs), *r)
    return res


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modelo", help="modelo de velocidades (.su do png2su ou .npy)")
    p.add_argument("--dx", type=float, required=True, help="tamanho do pixel em metros")
    p.add_argument("--f0", type=float, default=20.0, help="frequência de pico do Ricker (Hz)")
    p.add_argument("--tmax", type=float, default=1.0, help="duração do registro (s)")
    p.add_argument("--dt-out", type=float, default=0.001, help="amostragem do sismograma (s)")
    p.add_argument("--sx", required=True, help="fontes x em m: valor ou ini:fim:passo")
    p.add_argument("--sz", type=float, default=0.0, help="profundidade das fontes (m)")
    p.add_argument("--rx", required=True, help="receptores x em m: ini:fim:passo")
    p.add_argument("--rz", type=float, default=0.0, help="profundidade dos receptores (m)")
    p.add_argument("--nbl", type=int, default=20, help="espessura da PML (células)")
    p.add_argument("--so", type=int, default=8, help="ordem espacial (4, 8, 12...)")
    p.add_argument("--free-surface", action="store_true", help="superfície livre no topo (sem PML em cima)")
    p.add_argument("--procs", type=int, default=os.cpu_count(), help="tiros simulados ao mesmo tempo")
    p.add_argument("--out", default="tiros", help="pasta de saída")
    a = p.parse_args()

    v = load_model(a.modelo)
    cfg, info = plan(v, a.dx, a.f0, a.tmax, rng(a.sx), a.sz, rng(a.rx), a.rz, a.dt_out,
                     a.nbl, a.so, a.free_surface, a.out)
    print(f"Modelo: nx={info['nx']} (Y), nz={info['nz']} (Z), dx={a.dx} m -> "
          f"{info['width']:.0f} m x {info['depth']:.0f} m; {info['vmin']:.0f} a {info['vmax']:.0f} m/s")
    print(f"dt de simulação = {info['dt']*1e3:.4f} ms (estável <= {info['dt_stable']*1e3:.4f} ms), "
          f"{info['nt']} passos; PML = {a.nbl} células")
    for w in info["warnings"]:
        print("AVISO:", w)
    if info["errors"]:
        sys.exit("ERRO: " + " ".join(info["errors"]))
    print(f"{info['nshots']} tiro(s), {info['nrec']} receptores, {min(a.procs, info['nshots'])} processo(s)\n")
    t0 = time.time()
    res = run(cfg, a.procs, lambda d, n, path, sec: print(f"  [{d}/{n}] {path}  ({sec:.1f} s)", flush=True))
    print(f"\nPronto em {time.time() - t0:.1f} s. Ver um tiro:  suximage < {res[0][0]} perc=99 &")


if __name__ == "__main__":
    main()