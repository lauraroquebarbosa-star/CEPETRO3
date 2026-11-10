"""
png2su_gui.py - PNG → SU: modelo de velocidades e simulação acústica 2D.

Fluxo em 3 etapas:
  1. Modelo      abrir o PNG, dar a velocidade de cada camada e o tamanho do pixel
  2. Aquisição   fontes, receptores, wavelet e bordas, com checagens automáticas
  3. Simulação   rodar no Devito (PML + tiros em paralelo) e ver os sismogramas

USO:  python3 png2su_gui.py [imagem.png]
"""
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

import png2su_core as core
import simula_devito as sim

# ------------------------------------------------------------------ aparência
BG, CARD, BORDER = "#f3f5f8", "#ffffff", "#e2e6ec"
TEXT, MUTED, FAINT = "#1b2430", "#5f6b7a", "#98a2b0"
ACCENT, ACCENT_DARK, ACCENT_SOFT = "#2563eb", "#1d4ed8", "#e8effd"
OK, WARN, ERR = "#15803d", "#b45309", "#b91c1c"
SIDE, SIDE_HI, SIDE_TEXT, SIDE_MUTED = "#111827", "#1f2937", "#f9fafb", "#9ca3af"
CMAP = "viridis"


def pick_font(root, *names):
    fams = set(tkfont.families(root))
    for n in names:
        if n in fams:
            return n
    return "TkDefaultFont"


class Fonts:
    def __init__(self, root):
        ui = pick_font(root, "Inter", "Segoe UI", "Helvetica Neue", "Ubuntu", "DejaVu Sans")
        mono = pick_font(root, "JetBrains Mono", "DejaVu Sans Mono", "Consolas", "Courier")
        self.family = ui
        self.body = (ui, 10)
        self.small = (ui, 9)
        self.label = (ui, 10)
        self.strong = (ui, 10, "bold")
        self.h1 = (ui, 15, "bold")
        self.h2 = (ui, 12, "bold")
        self.step = (ui, 11, "bold")
        self.mono = (mono, 9)
        self.big = (ui, 11, "bold")


def style_matplotlib(family):
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 9, "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlecolor": TEXT, "axes.titlelocation": "left",
        "figure.facecolor": CARD, "axes.facecolor": CARD,
    })


def setup_style(root, F):
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure(".", background=BG, foreground=TEXT, font=F.body, bordercolor=BORDER,
                 lightcolor=BORDER, darkcolor=BORDER, troughcolor=BORDER, focuscolor=ACCENT)
    st.configure("Card.TFrame", background=CARD)
    st.configure("Card.TLabel", background=CARD, foreground=TEXT, font=F.body)
    st.configure("CardMuted.TLabel", background=CARD, foreground=MUTED, font=F.small)
    st.configure("CardTitle.TLabel", background=CARD, foreground=TEXT, font=F.h2)
    st.configure("Unit.TLabel", background=CARD, foreground=FAINT, font=F.small)
    st.configure("H1.TLabel", background=BG, foreground=TEXT, font=F.h1)
    st.configure("Sub.TLabel", background=BG, foreground=MUTED, font=F.body)
    st.configure("TEntry", fieldbackground=CARD, padding=(8, 5), bordercolor=BORDER,
                 lightcolor=BORDER, darkcolor=BORDER, insertcolor=TEXT)
    st.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)],
           darkcolor=[("focus", ACCENT)])
    st.configure("Bad.TEntry", fieldbackground="#fdf2f2", bordercolor=ERR, lightcolor=ERR, darkcolor=ERR)
    st.configure("TSpinbox", fieldbackground=CARD, padding=(6, 4), arrowsize=12)
    st.configure("TCombobox", fieldbackground=CARD, padding=(6, 4), arrowsize=12)
    st.configure("TButton", padding=(14, 7), background=CARD, foreground=TEXT,
                 bordercolor=BORDER, font=F.body, relief="flat")
    st.map("TButton", background=[("active", "#f0f3f7"), ("disabled", BG)],
           foreground=[("disabled", FAINT)])
    st.configure("Accent.TButton", background=ACCENT, foreground="white", bordercolor=ACCENT,
                 lightcolor=ACCENT, darkcolor=ACCENT, font=F.strong, padding=(18, 9))
    st.map("Accent.TButton", background=[("disabled", "#b7c6ef"), ("active", ACCENT_DARK)],
           bordercolor=[("active", ACCENT_DARK)], foreground=[("disabled", "white")])
    st.configure("Seg.TButton", padding=(12, 5), background=CARD, foreground=MUTED, font=F.small)
    st.configure("SegOn.TButton", padding=(12, 5), background=ACCENT_SOFT, foreground=ACCENT,
                 bordercolor=ACCENT, font=(F.family, 9, "bold"))
    st.map("SegOn.TButton", background=[("active", ACCENT_SOFT)])
    st.configure("TCheckbutton", background=CARD, foreground=TEXT, font=F.body)
    st.map("TCheckbutton", background=[("active", CARD)])
    st.configure("Accent.Horizontal.TProgressbar", background=ACCENT, troughcolor="#e9edf3",
                 bordercolor="#e9edf3", lightcolor=ACCENT, darkcolor=ACCENT, thickness=10)
    st.configure("TScale", background=CARD, troughcolor="#e9edf3")
    st.configure("Vertical.TScrollbar", background=BG, troughcolor=CARD, bordercolor=CARD,
                 arrowcolor=FAINT, gripcount=0)
    return st


# ------------------------------------------------------------------ peças de interface
class Card(tk.Frame):
    """Cartão branco com título opcional."""

    def __init__(self, parent, title=None, subtitle=None, F=None, **kw):
        super().__init__(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=BORDER, **kw)
        self.body = ttk.Frame(self, style="Card.TFrame", padding=(16, 12, 16, 14))
        self.body.pack(fill="both", expand=True)
        if title:
            ttk.Label(self.body, text=title, style="CardTitle.TLabel").grid(
                row=0, column=0, columnspan=4, sticky="w")
            if subtitle:
                ttk.Label(self.body, text=subtitle, style="CardMuted.TLabel", wraplength=330,
                          justify="left").grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))
        self.row = 2


def field(card, label, var, unit="", width=10, hint=None):
    """Linha de formulário: rótulo · campo · unidade (+ dica embaixo)."""
    r = card.row
    ttk.Label(card.body, text=label, style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=(10, 0))
    e = ttk.Entry(card.body, textvariable=var, width=width, justify="right")
    e.grid(row=r, column=1, sticky="e", padx=(12, 6), pady=(10, 0))
    ttk.Label(card.body, text=unit, style="Unit.TLabel", width=5).grid(row=r, column=2, sticky="w", pady=(10, 0))
    card.body.columnconfigure(0, weight=1)
    card.row += 1
    if hint:
        h = ttk.Label(card.body, text=hint, style="CardMuted.TLabel")
        h.grid(row=card.row, column=0, columnspan=3, sticky="w")
        card.row += 1
        return e, h
    return e, None


class StepButton(tk.Frame):
    def __init__(self, parent, n, title, subtitle, command, F):
        super().__init__(parent, bg=SIDE, cursor="hand2")
        self.n, self.F, self.command = n, F, command
        self.c = tk.Canvas(self, width=30, height=30, bg=SIDE, highlightthickness=0)
        self.c.grid(row=0, column=0, rowspan=2, padx=(18, 12), pady=10)
        self.t = tk.Label(self, text=title, bg=SIDE, fg=SIDE_TEXT, font=F.step, anchor="w")
        self.t.grid(row=0, column=1, sticky="sw", pady=(10, 0))
        self.s = tk.Label(self, text=subtitle, bg=SIDE, fg=SIDE_MUTED, font=F.small, anchor="w")
        self.s.grid(row=1, column=1, sticky="nw", pady=(0, 10))
        self.columnconfigure(1, weight=1)
        for w in (self, self.c, self.t, self.s):
            w.bind("<Button-1>", lambda e: self.command())
        self.set(False, False)

    def set(self, active, done, enabled=True):
        bg = SIDE_HI if active else SIDE
        for w in (self, self.c, self.t, self.s):
            w.configure(bg=bg)
        self.t.configure(fg=SIDE_TEXT if enabled else "#6b7280")
        self.c.delete("all")
        if done and not active:
            self.c.create_oval(3, 3, 27, 27, fill=OK, outline=OK)
            self.c.create_line(9, 15, 13.5, 19.5, 21, 10.5, fill="white", width=2.4,
                               capstyle="round", joinstyle="round")
        else:
            fill = ACCENT if active else bg
            outline = ACCENT if active else "#4b5563"
            self.c.create_oval(3, 3, 27, 27, fill=fill, outline=outline, width=2)
            self.c.create_text(15, 15, text=str(self.n), fill="white" if active else SIDE_MUTED,
                               font=(self.F.family, 10, "bold"))


class CheckList(ttk.Frame):
    """Lista de checagens com bolinha colorida (ok / aviso / erro)."""

    def __init__(self, parent, F):
        super().__init__(parent, style="Card.TFrame")
        self.F = F

    def show(self, items):
        for w in self.winfo_children():
            w.destroy()
        colors = {"ok": OK, "warn": WARN, "err": ERR}
        for i, (kind, text) in enumerate(items):
            dot = tk.Canvas(self, width=10, height=10, bg=CARD, highlightthickness=0)
            dot.create_oval(1, 1, 9, 9, fill=colors[kind], outline=colors[kind])
            dot.grid(row=i, column=0, sticky="n", pady=(5, 0), padx=(0, 8))
            ttk.Label(self, text=text, style="Card.TLabel", wraplength=560, justify="left",
                      foreground=TEXT if kind == "ok" else colors[kind]).grid(row=i, column=1, sticky="w", pady=2)


class Scrollable(ttk.Frame):
    """Coluna rolável (para o painel de controles)."""

    def __init__(self, parent, width=380):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, width=width)
        self.sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.win, width=e.width))
        self.canvas.configure(yscrollcommand=self.sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.sb.pack(side="right", fill="y")
        for seq, d in (("<Button-4>", -1), ("<Button-5>", 1)):
            self.inner.bind_all(seq, lambda e, d=d: self._wheel(e, d), add="+")
        self.inner.bind_all("<MouseWheel>", lambda e: self._wheel(e, -1 if e.delta > 0 else 1), add="+")

    def _wheel(self, e, d):
        w = e.widget
        while w is not None:
            if w is self.canvas:
                self.canvas.yview_scroll(d, "units")
                return
            w = getattr(w, "master", None)


class PlotPanel(tk.Frame):
    """Cartão com uma figura matplotlib."""

    def __init__(self, parent, F, figsize=(7, 5)):
        super().__init__(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=BORDER)
        self.top = ttk.Frame(self, style="Card.TFrame", padding=(16, 12, 16, 0))
        self.top.pack(fill="x")
        self.fig = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().configure(bg=CARD, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def empty(self, text):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color=FAINT, fontsize=11,
                transform=ax.transAxes)
        self.canvas.draw_idle()


# ------------------------------------------------------------------ aplicativo
class App(tk.Tk):
    def __init__(self, png=None):
        super().__init__()
        self.title("png2su - modelagem sísmica")
        self.geometry("1320x840")
        self.minsize(1180, 760)
        self.configure(bg=BG)
        self.F = Fonts(self)
        setup_style(self, self.F)
        style_matplotlib(self.F.family)

        # estado
        self.png_path = None
        self.rgb = None
        self.palette = None
        self.counts = None
        self.cls = None            # (a, b, alpha) por pixel
        self.model = None          # (Z, Y) km/s
        self.vel_vars = []
        self.vel_entries = []
        self.results = []
        self.q = queue.Queue()
        self.running = False
        self.has_devito = importlib.util.find_spec("devito") is not None
        self._pending = None

        self._build()
        self.goto(0)
        if png:
            self.after(100, lambda: self.load_png(png))

    # ---------------------------------------------------------- estrutura
    def _build(self):
        F = self.F
        side = tk.Frame(self, bg=SIDE, width=250)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="png2su", bg=SIDE, fg=SIDE_TEXT, font=(F.family, 16, "bold"),
                 anchor="w").pack(fill="x", padx=20, pady=(22, 0))
        tk.Label(side, text="Modelagem acústica 2D", bg=SIDE, fg=SIDE_MUTED, font=F.small,
                 anchor="w").pack(fill="x", padx=20, pady=(2, 18))
        self.steps = [
            StepButton(side, 1, "Modelo", "PNG, camadas e escala", lambda: self.goto(0), F),
            StepButton(side, 2, "Aquisição", "Fontes, receptores, bordas", lambda: self.goto(1), F),
            StepButton(side, 3, "Simulação", "Devito · PML · paralelo", lambda: self.goto(2), F),
        ]
        for s in self.steps:
            s.pack(fill="x")
        # resumo
        tk.Frame(side, bg="#273244", height=1).pack(fill="x", padx=20, pady=18)
        tk.Label(side, text="RESUMO", bg=SIDE, fg=SIDE_MUTED, font=(F.family, 8, "bold"),
                 anchor="w").pack(fill="x", padx=20)
        self.summary = tk.Label(side, text="Nenhum modelo carregado", bg=SIDE, fg=SIDE_TEXT,
                                font=F.small, justify="left", anchor="nw")
        self.summary.pack(fill="x", padx=20, pady=(6, 0))
        tk.Label(side, text="Devito: " + ("instalado" if self.has_devito else "não encontrado"),
                 bg=SIDE, fg=SIDE_MUTED if self.has_devito else "#fca5a5", font=F.small,
                 anchor="w").pack(side="bottom", fill="x", padx=20, pady=16)

        self.main = ttk.Frame(self, padding=(24, 20, 24, 18))
        self.main.pack(side="left", fill="both", expand=True)
        self.header_t = ttk.Label(self.main, style="H1.TLabel")
        self.header_t.pack(anchor="w")
        self.header_s = ttk.Label(self.main, style="Sub.TLabel")
        self.header_s.pack(anchor="w", pady=(2, 14))
        self.footer = ttk.Frame(self.main)
        self.footer.pack(side="bottom", fill="x", pady=(14, 0))
        self.pages_holder = ttk.Frame(self.main)
        self.pages_holder.pack(fill="both", expand=True)
        self.status = ttk.Label(self.footer, style="Sub.TLabel")
        self.status.pack(side="left")
        self.next_btn = ttk.Button(self.footer, style="Accent.TButton", command=self._next)
        self.next_btn.pack(side="right")
        self.back_btn = ttk.Button(self.footer, text="Voltar", command=lambda: self.goto(self.page - 1))
        self.back_btn.pack(side="right", padx=(0, 8))

        self.pages = [self._page_model(), self._page_acq(), self._page_sim()]

    def _two_columns(self):
        page = ttk.Frame(self.pages_holder)
        left = Scrollable(page, width=390)
        left.pack(side="left", fill="y")
        right = ttk.Frame(page)
        right.pack(side="left", fill="both", expand=True, padx=(18, 0))
        return page, left.inner, right

    # ---------------------------------------------------------- página 1: modelo
    def _page_model(self):
        F = self.F
        page, left, right = self._two_columns()

        c = Card(left, "Imagem", "Cada cor do PNG é uma camada. Bordas suavizadas são "
                 "tratadas como mistura das camadas vizinhas.")
        c.pack(fill="x")
        b = ttk.Frame(c.body, style="Card.TFrame")
        b.grid(row=c.row, column=0, columnspan=3, sticky="we", pady=(12, 0))
        ttk.Button(b, text="Abrir PNG…", command=self.choose_png).pack(side="left")
        self.file_lbl = ttk.Label(b, text="nenhum arquivo", style="CardMuted.TLabel")
        self.file_lbl.pack(side="left", padx=10)
        c.row += 1
        self.dx = tk.StringVar(value="5")
        self.dx_entry, self.dims_lbl = field(c, "Tamanho do pixel", self.dx, "m", hint=" ")
        self.dx.trace_add("write", lambda *_: self.schedule_refresh())

        c2 = Card(left, "Camadas", "Velocidade de propagação de cada cor.")
        c2.pack(fill="x", pady=(14, 0))
        self.layers = ttk.Frame(c2.body, style="Card.TFrame")
        self.layers.grid(row=c2.row, column=0, columnspan=3, sticky="we", pady=(8, 0))
        c2.row += 1
        bb = ttk.Frame(c2.body, style="Card.TFrame")
        bb.grid(row=c2.row, column=0, columnspan=3, sticky="we", pady=(12, 0))
        ttk.Button(bb, text="Carregar velocidades…", command=self.load_vels).pack(side="left")
        ttk.Button(bb, text="Salvar…", command=self.save_vels).pack(side="left", padx=6)
        self.mix_lbl = ttk.Label(c2.body, text="", style="CardMuted.TLabel", wraplength=330)
        self.mix_lbl.grid(row=c2.row + 1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        c3 = Card(left, "Exportar", "O modelo de velocidades no formato Seismic Unix, "
                  "para usar em outros programas.")
        c3.pack(fill="x", pady=(14, 0))
        self.export_btn = ttk.Button(c3.body, text="Salvar modelo .su…", command=self.export_su,
                                     state="disabled")
        self.export_btn.grid(row=c3.row, column=0, sticky="w", pady=(12, 0))

        self.plot1 = PlotPanel(right, F)
        self.plot1.pack(fill="both", expand=True)
        seg = ttk.Frame(self.plot1.top, style="Card.TFrame")
        seg.pack(side="right")
        self.view = tk.StringVar(value="vel")
        self.seg_btns = {}
        for key, txt in (("img", "Imagem"), ("vel", "Velocidades")):
            bt = ttk.Button(seg, text=txt, command=lambda k=key: self.set_view(k))
            bt.pack(side="left", padx=(0, 4))
            self.seg_btns[key] = bt
        ttk.Label(self.plot1.top, text="Modelo", style="CardTitle.TLabel").pack(side="left")
        self.set_view("vel", redraw=False)
        self.plot1.empty("Abra um PNG para começar")
        return page

    # ---------------------------------------------------------- página 2: aquisição
    def _page_acq(self):
        page, left, right = self._two_columns()
        V = lambda v: tk.StringVar(value=v)
        self.f0, self.tmax, self.dtout = V("20"), V("1.0"), V("1.0")
        self.sx0, self.sx1, self.sxd, self.sz = V("100"), V("900"), V("100"), V("10")
        self.rx0, self.rx1, self.rxd, self.rz = V("0"), V("1000"), V("5"), V("10")
        self.nbl, self.so = V("20"), V("8")
        self.free = tk.BooleanVar(value=False)

        c = Card(left, "Fonte sísmica", "Wavelet Ricker.")
        c.pack(fill="x")
        field(c, "Frequência de pico", self.f0, "Hz")
        field(c, "Duração do registro", self.tmax, "s")
        field(c, "Amostragem do sismograma", self.dtout, "ms")

        c = Card(left, "Fontes", "Posições ao longo da superfície do modelo.")
        c.pack(fill="x", pady=(14, 0))
        field(c, "Primeira posição x", self.sx0, "m")
        field(c, "Última posição x", self.sx1, "m")
        field(c, "Espaçamento", self.sxd, "m")
        _, self.src_hint = field(c, "Profundidade", self.sz, "m", hint=" ")

        c = Card(left, "Receptores")
        c.pack(fill="x", pady=(14, 0))
        field(c, "Primeira posição x", self.rx0, "m")
        field(c, "Última posição x", self.rx1, "m")
        field(c, "Espaçamento", self.rxd, "m")
        _, self.rec_hint = field(c, "Profundidade", self.rz, "m", hint=" ")

        c = Card(left, "Bordas e precisão", "A camada PML absorve as ondas que saem do modelo: "
                 "não é preciso estender a imagem.")
        c.pack(fill="x", pady=(14, 0))
        ttk.Checkbutton(c.body, text="Superfície livre no topo (reflete como a superfície real)",
                        variable=self.free, command=self.schedule_refresh).grid(
            row=c.row, column=0, columnspan=3, sticky="w", pady=(10, 0))
        c.row += 1
        field(c, "Espessura da PML", self.nbl, "células")
        r = c.row
        ttk.Label(c.body, text="Ordem espacial", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=(10, 0))
        cb = ttk.Combobox(c.body, textvariable=self.so, values=("4", "8", "12"), width=7,
                          state="readonly", justify="right")
        cb.grid(row=r, column=1, sticky="e", padx=(12, 6), pady=(10, 0))
        cb.bind("<<ComboboxSelected>>", lambda e: self.schedule_refresh())

        for v in (self.f0, self.tmax, self.dtout, self.sx0, self.sx1, self.sxd, self.sz,
                  self.rx0, self.rx1, self.rxd, self.rz, self.nbl):
            v.trace_add("write", lambda *_: self.schedule_refresh())

        # direita: gráfico + checagens
        cc = Card(right, "Checagens")
        cc.pack(side="bottom", fill="x", pady=(14, 0))
        self.plot2 = PlotPanel(right, self.F, figsize=(7, 4.2))
        self.plot2.pack(fill="both", expand=True)
        ttk.Label(self.plot2.top, text="Geometria da aquisição", style="CardTitle.TLabel").pack(side="left")
        self.checks = CheckList(cc.body, self.F)
        self.checks.grid(row=cc.row, column=0, sticky="we", pady=(8, 0))
        return page

    # ---------------------------------------------------------- página 3: simulação
    def _page_sim(self):
        page, left, right = self._two_columns()
        c = Card(left, "Execução", "Cada tiro é uma simulação independente; vários rodam "
                 "ao mesmo tempo, um por núcleo.")
        c.pack(fill="x")
        self.outdir = tk.StringVar(value=os.path.abspath("tiros"))
        r = c.row
        ttk.Label(c.body, text="Pasta de saída", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=(12, 0))
        ttk.Button(c.body, text="Escolher…", command=self.choose_out).grid(row=r, column=1, sticky="e", pady=(12, 0))
        self.out_lbl = ttk.Label(c.body, textvariable=self.outdir, style="CardMuted.TLabel", wraplength=330)
        self.out_lbl.grid(row=r + 1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        c.row = r + 2
        self.procs = tk.StringVar(value=str(os.cpu_count() or 1))
        r = c.row
        ttk.Label(c.body, text="Processos em paralelo", style="Card.TLabel").grid(row=r, column=0, sticky="w", pady=(12, 0))
        ttk.Spinbox(c.body, from_=1, to=max(1, os.cpu_count() or 1), textvariable=self.procs, width=6,
                    justify="right").grid(row=r, column=1, sticky="e", pady=(12, 0))
        c.row += 1
        self.run_btn = ttk.Button(c.body, text="Simular", style="Accent.TButton", command=self.run_sim)
        self.run_btn.grid(row=c.row, column=0, columnspan=3, sticky="we", pady=(18, 0))
        self.pbar = ttk.Progressbar(c.body, style="Accent.Horizontal.TProgressbar", mode="determinate")
        self.pbar.grid(row=c.row + 1, column=0, columnspan=3, sticky="we", pady=(14, 0))
        self.prog_lbl = ttk.Label(c.body, text="", style="CardMuted.TLabel")
        self.prog_lbl.grid(row=c.row + 2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        if not self.has_devito:
            ttk.Label(c.body, text="O Devito não está instalado. No terminal:\n"
                      "pip install --user devito", style="Card.TLabel", foreground=ERR,
                      justify="left").grid(row=c.row + 3, column=0, columnspan=3, sticky="w", pady=(10, 0))

        c = Card(left, "Registro")
        c.pack(fill="both", expand=True, pady=(14, 0))
        self.log = tk.Text(c.body, height=12, width=44, font=self.F.mono, bg="#f8fafc", fg=TEXT,
                           relief="flat", highlightthickness=1, highlightbackground=BORDER,
                           padx=8, pady=6, state="disabled", wrap="word")
        self.log.grid(row=c.row, column=0, columnspan=3, sticky="nswe", pady=(8, 0))

        self.plot3 = PlotPanel(right, self.F)
        self.plot3.pack(fill="both", expand=True)
        ttk.Label(self.plot3.top, text="Sismograma", style="CardTitle.TLabel").pack(side="left")
        ctl = ttk.Frame(self.plot3.top, style="Card.TFrame")
        ctl.pack(side="right")
        ttk.Label(ctl, text="Tiro", style="CardMuted.TLabel").pack(side="left", padx=(0, 6))
        self.shot = tk.StringVar()
        self.shot_cb = ttk.Combobox(ctl, textvariable=self.shot, width=22, state="readonly")
        self.shot_cb.pack(side="left")
        self.shot_cb.bind("<<ComboboxSelected>>", lambda e: self.show_gather())
        ttk.Label(ctl, text="Contraste", style="CardMuted.TLabel").pack(side="left", padx=(14, 6))
        self.clip = tk.DoubleVar(value=98.0)
        ttk.Scale(ctl, from_=90, to=99.9, variable=self.clip, length=110,
                  command=lambda v: self.show_gather()).pack(side="left")
        self.plot3.empty("Os sismogramas aparecem aqui depois da simulação")
        return page

    # ---------------------------------------------------------- navegação
    def goto(self, i):
        if i == 2 and self.model is None:
            i = 0 if self.model is None else 1
        if i == 1 and self.model is None:
            self.flash("Complete o modelo antes: abra o PNG e preencha todas as velocidades.")
            i = 0
        i = max(0, min(2, i))
        self.page = i
        for p in self.pages:
            p.pack_forget()
        self.pages[i].pack(fill="both", expand=True)
        titles = [("Modelo de velocidades", "Abra o PNG, informe a velocidade de cada camada e o tamanho do pixel."),
                  ("Aquisição", "Onde ficam as fontes e os receptores, e como a onda é gerada."),
                  ("Simulação", "Propagação acústica 2D no Devito, com borda absorvente PML.")]
        self.header_t.configure(text=titles[i][0])
        self.header_s.configure(text=titles[i][1])
        self.back_btn.configure(state="normal" if i > 0 else "disabled")
        self.next_btn.configure(text=["Próximo: Aquisição", "Próximo: Simulação", "Abrir pasta de saída"][i])
        if i == 1:
            self.refresh_acq()
        self._update_steps()

    def _next(self):
        if self.page < 2:
            self.goto(self.page + 1)
        else:
            path = self.outdir.get()
            if os.path.isdir(path):
                opener = "open" if sys.platform == "darwin" else "xdg-open"   # macOS / Linux
                os.system(f'{opener} "{path}" >/dev/null 2>&1 &')

    def _update_steps(self):
        done = [self.model is not None, self.model is not None and not self.acq_errors(), bool(self.results)]
        for i, s in enumerate(self.steps):
            s.set(i == self.page, done[i], enabled=(i == 0 or self.model is not None))

    def flash(self, msg, kind="warn"):
        self.status.configure(text=msg, foreground={"warn": WARN, "err": ERR, "ok": OK}.get(kind, MUTED))
        self.after(6000, lambda: self.status.configure(text=""))

    # ---------------------------------------------------------- modelo
    def choose_png(self):
        p = filedialog.askopenfilename(filetypes=[("Imagens PNG", "*.png"), ("Todos", "*")])
        if p:
            self.load_png(p)

    def load_png(self, path):
        try:
            rgb = core.load_image(path)
        except Exception as e:
            messagebox.showerror("Não foi possível abrir a imagem", str(e))
            return
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            pal, counts = core.detect_palette(rgb)
            self.cls = core.classify(rgb, pal)
            n_raw = len(core.find_colors(rgb)[0])
        finally:
            self.configure(cursor="")
        self.png_path, self.rgb, self.palette, self.counts = path, rgb, pal, counts
        self.file_lbl.configure(text=os.path.basename(path))
        self.mix_lbl.configure(text=(f"{n_raw - len(pal)} cores de borda (suavização) serão tratadas como "
                                     "mistura das camadas vizinhas, com média harmônica."
                                     if n_raw > len(pal) else "Imagem sem suavização nas bordas."))
        self._fill_layers()
        self.model = None
        self.results = []
        self.refresh_model()

    def _fill_layers(self):
        for w in self.layers.winfo_children():
            w.destroy()
        self.vel_vars, self.vel_entries = [], []
        total = self.counts.sum()
        hdr = [("", 0), ("Camada", 1), ("Área", 2), ("Velocidade", 3)]
        for t, col in hdr:
            ttk.Label(self.layers, text=t, style="CardMuted.TLabel").grid(row=0, column=col, sticky="w", padx=(0, 8))
        for i, ((r, g, b), n) in enumerate(zip(self.palette.tolist(), self.counts), start=1):
            sw = tk.Canvas(self.layers, width=26, height=18, bg=CARD, highlightthickness=0)
            sw.create_rectangle(1, 1, 25, 17, fill=f"#{r:02x}{g:02x}{b:02x}", outline=BORDER)
            sw.grid(row=i, column=0, pady=4, padx=(0, 8))
            ttk.Label(self.layers, text=f"Camada {i}", style="Card.TLabel").grid(row=i, column=1, sticky="w", padx=(0, 8))
            ttk.Label(self.layers, text=f"{100 * n / total:.1f}%", style="CardMuted.TLabel").grid(
                row=i, column=2, sticky="w", padx=(0, 8))
            var = tk.StringVar()
            e = ttk.Entry(self.layers, textvariable=var, width=8, justify="right")
            e.grid(row=i, column=3, sticky="e")
            ttk.Label(self.layers, text="km/s", style="Unit.TLabel").grid(row=i, column=4, sticky="w", padx=(6, 0))
            var.trace_add("write", lambda *_: self.schedule_refresh())
            self.vel_vars.append(var)
            self.vel_entries.append(e)
        self.layers.columnconfigure(1, weight=1)
        if self.vel_entries:
            self.vel_entries[0].focus_set()

    def read_vels(self):
        """Lista de velocidades (km/s) ou None; marca em vermelho os campos inválidos."""
        out, ok = [], True
        for var, e in zip(self.vel_vars, self.vel_entries):
            txt = var.get().strip().replace(",", ".")
            try:
                v = float(txt)
                good = 0.05 <= v <= 20
            except ValueError:
                v, good = None, False
            e.configure(style="TEntry" if (good or not txt) else "Bad.TEntry")
            ok &= good
            out.append(v)
        return out if ok and out else None

    def read_float(self, var, default=None):
        try:
            return float(var.get().strip().replace(",", "."))
        except ValueError:
            return default

    def schedule_refresh(self):
        if self._pending:
            self.after_cancel(self._pending)
        self._pending = self.after(250, self._refresh_now)

    def _refresh_now(self):
        self._pending = None
        if self.page == 0:
            self.refresh_model()
        elif self.page == 1:
            self.refresh_acq()

    def refresh_model(self):
        if self.rgb is None:
            return
        dx = self.read_float(self.dx)
        nz, ny = self.rgb.shape[:2]
        if dx and dx > 0:
            self.dims_lbl.configure(text=f"Modelo: {(ny - 1) * dx:,.0f} m × {(nz - 1) * dx:,.0f} m  "
                                    f"({ny} × {nz} pixels)".replace(",", "."))
            self.dx_entry.configure(style="TEntry")
        else:
            self.dims_lbl.configure(text="Informe o tamanho do pixel em metros.")
            self.dx_entry.configure(style="Bad.TEntry")
        vels = self.read_vels()
        if vels is not None and dx and dx > 0:
            a, b, al = self.cls
            self.model = core.effective_velocity(a, b, al, vels)
            self.export_btn.configure(state="normal")
            self.status.configure(text="")
        else:
            self.model = None
            self.export_btn.configure(state="disabled")
        self.draw_model()
        self.update_summary()
        self._update_steps()

    def set_view(self, key, redraw=True):
        self.view.set(key)
        for k, b in self.seg_btns.items():
            b.configure(style="SegOn.TButton" if k == key else "Seg.TButton")
        if redraw:
            self.draw_model()

    def _extent(self):
        dx = self.read_float(self.dx, 1.0) or 1.0
        nz, ny = self.rgb.shape[:2]
        return [0, (ny - 1) * dx, (nz - 1) * dx, 0]

    def draw_model(self):
        P = self.plot1
        if self.rgb is None:
            return P.empty("Abra um PNG para começar")
        P.fig.clear()
        ax = P.fig.add_subplot(111)
        ext = self._extent()
        if self.view.get() == "img" or self.model is None:
            ax.imshow(self.rgb, extent=ext, interpolation="nearest", aspect="equal")
            ax.set_title("Imagem original" if self.view.get() == "img" else
                         "Imagem original  ·  preencha as velocidades para ver o modelo")
        else:
            im = ax.imshow(self.model, extent=ext, cmap=CMAP, interpolation="nearest", aspect="equal")
            cb = P.fig.colorbar(im, ax=ax, fraction=0.035, pad=0.05)
            cb.set_label("Velocidade (km/s)")
            cb.outline.set_edgecolor(BORDER)
            ax.set_title(f"Velocidades  ·  {self.model.min():.2f} a {self.model.max():.2f} km/s")
        ax.set_xlabel("Distância x (m)")
        ax.set_ylabel("Profundidade z (m)")
        P.fig.tight_layout()
        P.canvas.draw_idle()

    def update_summary(self):
        if self.rgb is None:
            return
        nz, ny = self.rgb.shape[:2]
        dx = self.read_float(self.dx)
        lines = [os.path.basename(self.png_path), f"{ny} × {nz} pixels · {len(self.palette)} camadas"]
        if dx:
            lines.append(f"{(ny - 1) * dx:,.0f} × {(nz - 1) * dx:,.0f} m".replace(",", "."))
        if self.model is not None:
            lines.append(f"{self.model.min():.2f} a {self.model.max():.2f} km/s")
        if self.results:
            lines.append(f"{len(self.results)} tiros simulados")
        self.summary.configure(text="\n".join(lines))

    def save_vels(self):
        if self.palette is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            data = {f"{r},{g},{b}": v.get() for (r, g, b), v in zip(self.palette.tolist(), self.vel_vars)}
            data["_pixel_m"] = self.dx.get()
            with open(p, "w") as f:
                json.dump(data, f, indent=2)
            self.flash(f"Velocidades salvas em {os.path.basename(p)}", "ok")

    def load_vels(self):
        if self.palette is None:
            return
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        with open(p) as f:
            data = json.load(f)
        n = 0
        for (r, g, b), v in zip(self.palette.tolist(), self.vel_vars):
            if f"{r},{g},{b}" in data:
                v.set(str(data[f"{r},{g},{b}"]))
                n += 1
        if "_pixel_m" in data:
            self.dx.set(str(data["_pixel_m"]))
        self.flash(f"{n} de {len(self.vel_vars)} velocidades carregadas.", "ok")

    def export_su(self):
        if self.model is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".su", filetypes=[("Seismic Unix", "*.su")],
                                         initialfile=os.path.basename(core.default_output(self.png_path)))
        if not p:
            return
        try:
            header = core.write_su(self.model, p)
            self.flash(f"Modelo salvo em {os.path.basename(p)}", "ok")
            self.write_log(f"Modelo exportado: {p}\n{header}")
        except Exception as e:
            messagebox.showerror("Erro ao gerar o .su", str(e))

    # ---------------------------------------------------------- aquisição
    def acq_plan(self):
        """(cfg, info) com os valores da tela, ou (None, info com erros)."""
        if self.model is None:
            return None, {"errors": ["Modelo incompleto."], "warnings": []}
        g = self.read_float
        vals = dict(dx=g(self.dx), f0=g(self.f0), tmax=g(self.tmax), dtout=g(self.dtout),
                    sx0=g(self.sx0), sx1=g(self.sx1), sxd=g(self.sxd), sz=g(self.sz),
                    rx0=g(self.rx0), rx1=g(self.rx1), rxd=g(self.rxd), rz=g(self.rz), nbl=g(self.nbl))
        bad = [k for k, v in vals.items() if v is None]
        if bad:
            return None, {"errors": ["Preencha todos os campos com números."], "warnings": []}
        if min(vals["f0"], vals["tmax"], vals["dtout"], vals["dx"]) <= 0 or vals["sxd"] <= 0 or vals["rxd"] <= 0:
            return None, {"errors": ["Frequência, duração, amostragem e espaçamentos devem ser positivos."],
                          "warnings": []}
        sx = np.arange(vals["sx0"], vals["sx1"] + vals["sxd"] / 2, vals["sxd"])
        rx = np.arange(vals["rx0"], vals["rx1"] + vals["rxd"] / 2, vals["rxd"])
        v = self.model.T.astype(np.float32) * 1000.0
        return sim.plan(v, vals["dx"], vals["f0"], vals["tmax"], sx, vals["sz"], rx, vals["rz"],
                        dt_out=vals["dtout"] / 1000.0, nbl=int(vals["nbl"]), so=int(self.so.get()),
                        free_surface=self.free.get(), out=self.outdir.get())

    def acq_errors(self):
        try:
            return self.acq_plan()[1]["errors"]
        except Exception as e:
            return [str(e)]

    def refresh_acq(self):
        cfg, info = self.acq_plan()
        items = [("err", e) for e in info["errors"]] + [("warn", w) for w in info["warnings"]]
        if cfg is not None:
            ok = []
            if not any("Pixel" in w for w in info["warnings"]):
                ok.append(("ok", f"Malha adequada: pixel ≤ {info['dx_max']:.2f} m para "
                                 f"{self.f0.get()} Hz (sem dispersão numérica)."))
            ok.append(("ok", f"Passo de tempo estável escolhido: {info['dt'] * 1e3:.3f} ms "
                             f"({info['nt']:,} passos).".replace(",", ".")))
            if not info["errors"]:
                ok.append(("ok", f"{info['nshots']} tiro(s) × {info['nrec']} receptores, "
                                 f"{info['nt_out']} amostras por traço."))
                ok.append(("ok", f"PML de {int(self.read_float(self.nbl))} células em "
                                 f"{'3 lados (topo livre)' if self.free.get() else '4 lados'}: "
                                 "sem extensão da imagem."))
            items = ok + items
            self.src_hint.configure(text=f"{info['nshots']} fonte(s)")
            self.rec_hint.configure(text=f"{info['nrec']} receptores")
        self.checks.show(items)
        self.draw_acq(cfg, info)
        self._update_steps()

    def draw_acq(self, cfg, info):
        P = self.plot2
        if self.model is None:
            return P.empty("Complete o modelo na etapa 1")
        P.fig.clear()
        ax = P.fig.add_subplot(111)
        ext = self._extent()
        im = ax.imshow(self.model, extent=ext, cmap=CMAP, interpolation="nearest", aspect="equal", alpha=0.9)
        cb = P.fig.colorbar(im, ax=ax, fraction=0.035, pad=0.05)
        cb.set_label("Velocidade (km/s)")
        cb.outline.set_edgecolor(BORDER)
        if cfg is not None:
            dx, nbl = cfg["dx"], cfg["nbl"]
            W, D = ext[1], ext[2]
            pml = nbl * dx
            top = 0 if cfg["free_surface"] else pml
            ax.add_patch(Rectangle((-pml, -top), W + 2 * pml, D + pml + top, fill=False,
                                   edgecolor=FAINT, linestyle=(0, (4, 3)), linewidth=1))
            ax.text(W + pml, D + pml, " PML", color=FAINT, fontsize=8, va="bottom", ha="right")
            ax.scatter(cfg["rx"], cfg["rz"], marker="v", s=12, color="#f8fafc", edgecolors="#334155",
                       linewidths=0.6, zorder=3, label="Receptores")
            ax.scatter(cfg["sx"], np.full(len(cfg["sx"]), cfg["sz"]), marker="*", s=110, color="#ef4444",
                       edgecolors="white", linewidths=0.8, zorder=4, label="Fontes")
            ax.set_xlim(-pml * 1.05, W + pml * 1.05)
            ax.set_ylim(D + pml * 1.05, -top * 1.05 - (0.04 * D))
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, fontsize=8, frameon=False)
        ax.set_title("Fontes e receptores sobre o modelo")
        ax.set_xlabel("Distância x (m)")
        ax.set_ylabel("Profundidade z (m)")
        P.fig.tight_layout()
        P.canvas.draw_idle()

    # ---------------------------------------------------------- simulação
    def choose_out(self):
        p = filedialog.askdirectory(initialdir=self.outdir.get())
        if p:
            self.outdir.set(p)

    def write_log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def run_sim(self):
        if self.running:
            return
        if not self.has_devito:
            messagebox.showerror("Devito não encontrado", "Instale com:\n\npip install --user devito")
            return
        cfg, info = self.acq_plan()
        if cfg is None or info["errors"]:
            messagebox.showerror("Revise a aquisição", "\n".join(info["errors"]))
            self.goto(1)
            return
        try:
            procs = max(1, int(self.procs.get()))
        except ValueError:
            procs = 1
        self.running = True
        self.results = []
        self.run_btn.configure(state="disabled", text="Simulando…")
        self.pbar.configure(maximum=info["nshots"], value=0)
        self.prog_lbl.configure(text=f"0 de {info['nshots']} tiros")
        self.write_log(f"{time.strftime('%H:%M:%S')}  {info['nshots']} tiro(s), {info['nrec']} receptores, "
                       f"dt = {info['dt'] * 1e3:.3f} ms, {procs} processo(s)")
        self.t0 = time.time()

        def work():
            try:
                sim.run(cfg, procs, lambda d, n, path, sec: self.q.put(("shot", d, n, path, sec)))
                self.q.put(("done",))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.q.put(("error", f"{type(e).__name__}: {e}"))
        threading.Thread(target=work, daemon=True).start()
        self.after(150, self._poll)

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "shot":
                    _, d, n, path, sec = msg
                    self.results.append(path)
                    self.pbar.configure(value=d)
                    el = time.time() - self.t0
                    eta = el / d * (n - d)
                    self.prog_lbl.configure(text=f"{d} de {n} tiros · {el:.0f} s"
                                            + (f" · faltam ~{eta:.0f} s" if d < n else ""))
                    self.write_log(f"  [{d}/{n}] {os.path.basename(path)}  ({sec:.1f} s)")
                    names = [os.path.basename(p) for p in self.results]
                    self.shot_cb.configure(values=names)
                    if d == 1:
                        self.shot.set(names[0])
                        self.show_gather()
                elif msg[0] == "done":
                    self._finish(f"Pronto em {time.time() - self.t0:.1f} s · arquivos em {self.outdir.get()}", "ok")
                    return
                elif msg[0] == "error":
                    self.write_log("ERRO: " + msg[1])
                    self._finish("A simulação falhou: veja o registro.", "err")
                    messagebox.showerror("Erro na simulação", msg[1])
                    return
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def _finish(self, text, kind):
        self.running = False
        self.run_btn.configure(state="normal", text="Simular")
        self.write_log(text)
        self.flash(text, kind)
        self.update_summary()
        self._update_steps()

    def show_gather(self):
        P = self.plot3
        name = self.shot.get()
        path = next((p for p in self.results if os.path.basename(p) == name), None)
        if not path or not os.path.exists(path):
            return
        raw = np.fromfile(path, dtype=np.uint8)
        ns = int(raw[114:116].view(np.uint16)[0])
        dt = int(raw[116:118].view(np.uint16)[0]) * 1e-6
        tr = raw.reshape(-1, 240 + 4 * ns)
        data = tr[:, 240:].copy().view(np.float32)
        gx = tr[:, 80:84].copy().view(np.int32).ravel()
        sx = int(tr[0, 72:76].copy().view(np.int32)[0])
        c = np.percentile(np.abs(data), self.clip.get()) or 1.0
        P.fig.clear()
        ax = P.fig.add_subplot(111)
        ax.imshow(data.T, aspect="auto", cmap="gray_r", vmin=-c, vmax=c, interpolation="bilinear",
                  extent=[gx.min(), gx.max(), (ns - 1) * dt, 0])
        ax.axvline(sx, color="#ef4444", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{name}  ·  fonte em x = {sx} m")
        ax.set_xlabel("Posição do receptor x (m)")
        ax.set_ylabel("Tempo (s)")
        P.fig.tight_layout()
        P.canvas.draw_idle()


if __name__ == "__main__":
    mp.freeze_support()
    App(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()