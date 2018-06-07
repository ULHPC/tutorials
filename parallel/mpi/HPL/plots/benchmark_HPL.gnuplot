################################################################################
# benchmark_HPL.gnuplot - 
#    Configuration file for Gnuplot (see http://www.gnuplot.info)
# Time-stamp: <Ven 2013-11-08 10:50 svarrette>
#
# Copyright (c) 2013 Sebastien Varrette <Sebastien.Varrette@uni.lu>
#               http://varrette.gforge.uni.lu
#
# More infos and tips on Gnuplot: 
#    - http://t16web.lanl.gov/Kawano/gnuplot/index-e.html
#    - run gnuplot the type 'help gnuplot_command'
#   
# To run this script : gnuplot benchmark_network.gnuplot
################################################################################

set encoding iso_8859_15
#########################################################################################
# Formats de sortie
# - Par défaut : sur X11
# - LaTeX (permet d'utiliser le format LaTeX $expr$ pour les formules)
#set terminal latex
#set output "outputfile.tex"
# - EPS
set terminal postscript eps enhanced color
#set output "outputfile.eps"
# - PDF
#set terminal pdf enhanced
#########################################################################################

set border 3
set xtics nomirror
set ytics nomirror
set grid
set size 0.8,0.8
#set xlabel "N (Problem size)" 
set ylabel "HPL R_{peak} [GFlops]"
#set yformat 

set key outside right #top left

set style data histogram
set style histogram cluster gap 3
set samples 11
set style fill transparent solid 0.5 noborder
#set boxwidth 0.9
#set multiplot

set bmargin at screen 0.10

# every I:J:K:L:M:N
# I Line increment
# J Data block increment
# K The first line
# L The first data block
# M The last line
# N The last data block


################################################################################
#   Performances with High-Performance Linpack (HPL) with Intel Suite 
################################################################################
set output "benchmark_HPL_2H_icc_imkl.eps"
set title  "HPLinpack 2.1  -- 2 nodes -- Intel MKL (icc and iMPI)"


#96 112 128 144 168 192 224 256           NBs
# 1 2  4            Ps
#32 16 8            Qs

plot \
     "< grep WR results_icc_imkl.dat" \
     every    8    using 7:xticlabel("N=" . stringcolumn(2) . "\nPxQ = " . stringcolumn(4) . "x" . stringcolumn(5)) title "NB=96", \
     "" every 8::1 using 7         title "NB=112", \
     "" every 8::2 using 7         title "NB=128", \
     "" every 8::3 using 7         title "NB=144", \
     "" every 8::4 using 7         title "NB=168", \
     "" every 8::5 using 7         title "NB=192", \
     "" every 8::6 using 7         title "NB=224", \
     "" every 8::7 using 7         title "NB=256"


################################################################################
#   Performances on HPL with GCC and GotoBLAS2 and Open MPI
################################################################################
set output "benchmark_HPL_2H_gcc_gotoblas2.eps"
set title  "HPLinpack 2.1  -- 2 nodes -- GotoBlas2 (GCC and OpenMPI)"

plot \
     "< grep WR results_gcc_gotoblas2.dat" \
     every    8    using 7:xticlabel("N=" . stringcolumn(2) . "\nPxQ = " . stringcolumn(4) . "x" . stringcolumn(5)) title "NB=96", \
     "" every 8::1 using 7         title "NB=112", \
     "" every 8::2 using 7         title "NB=128", \
     "" every 8::3 using 7         title "NB=144", \
     "" every 8::4 using 7         title "NB=168", \
     "" every 8::5 using 7         title "NB=192", \
     "" every 8::6 using 7         title "NB=224", \
     "" every 8::7 using 7         title "NB=256"


################################################################################
#   Performances on HPL with GCC and ATLAS and MVAPICH2
################################################################################
set output "benchmark_HPL_2H_gcc_atlas.eps"
set title  "HPLinpack 2.1  -- 2 nodes -- ATLAS (GCC and MVAPICH2)"

plot \
     "< grep WR results_gcc_atlas.dat" \
     every    8    using 7:xticlabel("N=" . stringcolumn(2) . "\nPxQ = " . stringcolumn(4) . "x" . stringcolumn(5)) title "NB=96", \
     "" every 8::1 using 7         title "NB=112", \
     "" every 8::2 using 7         title "NB=128", \
     "" every 8::3 using 7         title "NB=144", \
     "" every 8::4 using 7         title "NB=168", \
     "" every 8::5 using 7         title "NB=192", \
     "" every 8::6 using 7         title "NB=224", \
     "" every 8::7 using 7         title "NB=256"



################################################################################
#   Comparing all HPL Performances 
################################################################################
set output "benchmark_HPL_2H.eps"
set key outside top center 
#set xlabel "NB (Block size)"

set style histogram clustered gap 1 title #offset 2,0.25
set style fill solid border
set boxwidth 0.95
set size 1.2,0.6


#set style histogram  cluster

set style line 1 lt 1 lc rgb "gray"
set style line 2 lt 1 lc rgb "green"
set style line 3 lt 1 lc rgb "blue"
set style line 4 lt 1 lc rgb "red"
set style line 5 lt 1 lc rgb "black"

set xtic offset 0,0.5

set title  "HPLinpack 2.1  -- 2 nodes - N=73497"
plot \
     newhistogram "PxQ=1x32", \
      "< grep WR results_icc_imkl.dat      | head -n 8" using 7:xtic(3) ls 1 title "Intel MKL    (icc + impi)", \
      "< grep WR results_gcc_gotoblas2.dat | head -n 8" using 7:xtic(3) ls 2 title "GotoBLAS2 (gcc + OpenMPI)", \
      "< grep WR results_gcc_atlas.dat     | head -n 8" using 7:xtic(3) ls 3 title "ATLAS    (gcc + MVAPICH2)", \
     newhistogram "PxQ=2x16", \
      "< grep WR results_icc_imkl.dat      | sed -n '9,16p'" using 7:xtic(3) ls 1 notitle, \
      "< grep WR results_gcc_gotoblas2.dat | sed -n '9,16p'" using 7:xtic(3) ls 2 notitle, \
      "< grep WR results_gcc_atlas.dat     | sed -n '9,16p'" using 7:xtic(3) ls 3 notitle, \
     newhistogram "PxQ=4x8", \
      "< grep WR results_icc_imkl.dat      | tail -n 8" using 7:xtic(3) ls 1 notitle, \
      "< grep WR results_gcc_gotoblas2.dat | tail -n 8" using 7:xtic(3) ls 2 notitle, \
      "< grep WR results_gcc_atlas.dat     | tail -n 8" using 7:xtic(3) ls 3 notitle




# 
#i#################
# Gnuplot Infos:
# 
# [Tab. 1] Liste des fonctions reconnues par Gnuplot :
# ----------------------------------------------------
#   abs    valeur absolue
#   acos   arc cosinus
#   asin   arc sinus
#   atan   arc tangente
#   cos    cosinus
#   exp    exponentiel
#   int    renvoi la partie  entière de son argument
#   log    logarithme
#   log10  logarithme en base 10
#   rand   random (renvoi un nombre entre 0 et 1)
#   real   partie real
#   sgn    renvoi 1 si l'argument est positif, 0 s'il
#          est nulle, et -1 s'il est négatif
#   sin    sinus
#   sqrt   racine carré
#   tan    tangente
#
# [Tab. 2] Operateurs reconnues par Gnuplot :
# -------------------------------------------
#    Symbole      Exemple         Explication
#     **           a**b           exponentiation
#     *            a*b            multiplication
#     /            a/b            division
#     %            a%b            modulo
#     +            a+b            addition
#     -            a-b            soustraction
#     ==           a==b           égalité
#     !=           a!=b           inégalité
#     &            a&b            ET
#     ^            a^b            OU exclusif
#     |            a|b            OU inclusif
#     &&           a&&b           ET logique
#     ||           a||b           OU logique
#     ?:           a?b:c          opération ternaire
#
# [Tab. 3] Liste des formats reconnus (instructions 'set format') :
# -----------------------------------------------------------------
#       Format       Explanation
#       %f           floating point notation
#       %e or %E     exponential notation; an "e" or "E" before the power
#       %g or %G     the shorter of %e (or %E) and %f
#       %x or %X     hex
#       %o or %O     octal
#       %t           mantissa to base 10
#       %l           mantissa to base of current logscale
#       %s           mantissa to base of current logscale; scientific power
#       %T           power to base 10
#       %L           power to base of current logscale
#       %S           scientific power
#       %c           character replacement for scientific power
#       %P           multiple of pi
#
# [Tab. 4] Valeur attribuée aux bordure (instruction 'set border <sum>') :
# ------------------------------------------------------------------------
#        Bit   axe affiché      
#         1     bas         (x ou x1)
#         2     gauche      (y ou y1 )
#         4     haut        (x2)
#         8     droit       (y2)     
#
# [Tab. 5] Affichage des lettres grecs en mode Postcript Ex: {/Symbol a} => \alpha
# +------------------------+--------------------+
# |  ALPHABET 	SYMBOL     | alphabet 	symbol  |
# +------------------------+--------------------+
#	A 	Alpha 	   | 	a 	alpha	|
#	B 	Beta 	   |  	b 	beta 	|
#	C 	Chi 	   |  	c 	chi 	|
#	D 	Delta 	   |  	d 	delta 	|
#	E 	Epsilon    |  	e 	epsilon |
#	F 	Phi  	   |  	f 	phi 	|
#	G 	Gamma 	   | 	g 	gamma 	|
#	H 	Eta 	   |  	h 	eta 	|
#	I 	iota 	   | 	i 	iota 	|
#	K 	Kappa 	   | 	k 	kappa 	|
#	L 	Lambda 	   |  	l 	lambda 	|
#	M 	Mu 	   | 	m 	mu 	|
#	N     	Nu         |  	n       nu   	|
#	O     	Omicron    |    o       omicron |
#	P     	Pi   	   |    p       pi	|
#	Q     	Theta 	   |  	q       theta	|
#	R     	Rho  	   |   	r       rho     |
#	S     	Sigma 	   | 	s       sigma	|
#	T     	Tau 	   | 	t       tau	|
#	U     	Upsilon	   |	u       upsilon	|
#	W     	Omega 	   |	w       omega	|
#	X     	Xi 	   |	x       xi	|
#	Y     	Psi 	   |	y       psi	|
#	Z     	Zeta 	   |	z       zeta	|




