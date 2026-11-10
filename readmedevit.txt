SIMULAÇÃO COM DEVITO A PARTIR DO PNG2SU
=======================================

1) INSTALAR (uma vez, sem precisar de administrador)

   pip install --user devito

   Precisa de um compilador C (gcc). Teste:  python3 -c "import devito; print(devito.__version__)"

2) PELA INTERFACE (recomendado)

   python3 png2su_gui.py

   Etapa 1 - Modelo:     abrir o PNG, velocidade de cada camada, tamanho do pixel
   Etapa 2 - Aquisição:  fontes, receptores, frequência, bordas (checagens automáticas)
   Etapa 3 - Simulação:  Simular -> um .su por tiro na pasta escolhida, com visualização

   Não existe mais extensão da imagem: a PML do simulador absorve as bordas.
   Os 3 arquivos .py (png2su_core, simula_devito, png2su_gui) ficam na mesma pasta.

3) OU PELO TERMINAL

   python3 png2su.py modelo.png        (gera modelo.su; não pergunta mais a extensão)
   e depois:

   python3 simula_devito.py modelo.su --dx 5 --f0 20 --tmax 1.0 \
       --sx 100:900:100 --sz 10 --rx 0:1000:5 --rz 10 --out tiros

   --dx      tamanho do pixel em metros (obrigatório)
   --f0      frequência de pico do Ricker em Hz (padrão 20)
   --tmax    duração do registro em segundos (padrão 1.0)
   --dt-out  amostragem do sismograma em segundos (padrão 0.001)
   --sx      posição x das fontes em metros: um valor, ou inicio:fim:passo
   --sz      profundidade das fontes em metros
   --rx      posição x dos receptores em metros: inicio:fim:passo
   --rz      profundidade dos receptores em metros
   --nbl     espessura da PML em células (padrão 20; 30 para mais precisão)
   --so      ordem espacial das diferenças finitas (padrão 8)
   --free-surface   superfície livre no topo (PML só nos outros 3 lados)
   --procs   quantos tiros rodar ao mesmo tempo (padrão: todos os núcleos)
   --out     pasta de saída

   Saída: um arquivo .su por tiro (tiro_001_sx100.su, ...), um traço por receptor,
   com sx, gx, offset, ns e dt preenchidos no cabeçalho.

4) VER O RESULTADO

   suximage < tiros/tiro_001_sx100.su perc=99 &

AVISOS QUE O SCRIPT PODE MOSTRAR

   "AVISO: dx=... > ... (dispersão numérica)": o pixel é grande demais para a
   frequência escolhida. Use um PNG com mais resolução, um f0 menor ou --so 12.

VALIDAÇÃO (modelo de 2 camadas, 1 km x 1 km, dx = 5 m, f0 = 20 Hz, T = 0,8 s)

   PML 10 células: erro 0,49%    custo 1,21x
   PML 20 células: erro 0,08%    custo 1,44x   <- padrão
   PML 30 células: erro 0,02%    custo 1,69x
   (erro contra uma simulação de referência sem nenhuma borda ao alcance)

   2 tiros em 2 processos: resultado idêntico, bit a bit, ao da execução em série.

MÉTODO

   Equação acústica de 1ª ordem (velocidade-pressão), malha intercalada,
   PML de campo dividido (Berenger, 1994), densidade constante.