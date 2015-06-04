################################################################################
# benchmark_OSU.gnuplot - 
#    Configuration file for Gnuplot (see http://www.gnuplot.info)
# Time-stamp: <Mar 2013-11-12 11:56 svarrette>
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
set terminal postscript eps enhanced #color
#set output "outputfile.eps"
# - PDF
#set terminal pdf enhanced
#########################################################################################

#########################################################################################
# Gestion de l'affichage des axes, des legendes etc....
#


# -- Gestion des axes --
set title  "OSU One Sided MPI Get latency Test v4.3"
set xlabel "Packet size (bits) - LOGSCALE"    
set format x "10^%T"
set ylabel "Latency ({/Symbol m}s) - LOGSCALE - the LOWER the better"    # Label Axe Y (avec 2 axes: set y{1|2}label)
set border 3           # format d'affichage des bordures (Valeur du paramètre: voir Tab.4 plus bas)
#set xrange [1:4194304]    # Intervalle Axe X 
#set yrange [1:100000]     # Intervalle Axe Y
set logscale x        # Echelle logarithmique pour l'axe des x (set logscale <axes> <base>)
set logscale y       # Echelle logarithmique en base 2 pour l'axe des y
set xtics nomirror    # Pas de reflexion de l'axe X en haut
set ytics nomirror    # Pas de reflexion de l'axe Y à droite
set grid              # affichage d'une grille pour les graduations majeures
set size 0.8,0.8

# -- Positionnement/affichage des legendes des courbes --
set key bottom   # Placé les légendes en bas à gauche

#########################################################################################


# set style line <index> {{linetype  | lt} <line_type> | <colorspec>}
#                               {{linecolor | lc} <colorspec>}
#                               {{linewidth | lw} <line_width>}
#                               {{pointtype | pt} <point_type>}
#                               {{pointsize | ps} <point_size>}
#                               {{pointinterval | pi} <interval>}
#                               {palette}

#set style line 1 lt 2 lc rgb "red" lw 3
# set style line 1 lw 2 pt 2 linecolor rgb "blue"
# set style line 2 lw 2 pt 1 linecolor rgb "cyan"
# set style line 3 lw 2 pt 3 linecolor rgb "red"
# set style line 4 lw 2 pt 4 linecolor rgb "orange"
#set style line 5 lt 2 lc rgb "black" lw 3
# set style line 6 lt 5 lc rgb "magenta" lw 4


##########################################################
#   Performances with OSU Micro-Benchmarks Latency Test  #
##########################################################
set output "benchmark_OSU_2H_latency.eps"

plot \
     "results_latency_impi.dat"     using 1:2 title "Intel MPI" with linespoints, \
     "results_latency_openmpi.dat"  using 1:2 title "OpenMPI"   with linespoints, \
     "results_latency_mvapich2.dat" using 1:2 title "MVAPICH2"  with linespoints


############################################################
#   Performances with OSU Micro-Benchmarks Bandwidth Test  #
############################################################
set output "benchmark_OSU_2H_bandwidth.eps"

set title  "OSU MPI One Sided MPI Get Bandwidth Test v4.3"  
set ylabel "Bandwidth (MB/s) - LOGSCALE - the HIGHER the better"

# set yrange [1:10000] 

# 1 GbE max 1 Gb/s
max_bw_gbe(x)=1000/8
# 10 GbE max 10 Gb/s
max_bw_tengbe(x)=10000/8
# IB QDR max 40 Gb/s
max_bw_ibqdr(x)=40000/8

plot \
     max_bw_ibqdr(x)           title "IB QDR Theoretical Max" lw 4 with lines, \
     max_bw_gbe(x)             title "1 GbE  Theoretical Max" lw 2 with lines, \
     "results_bw_impi.dat"     using 1:2 title "Intel MPI" with linespoints, \
     "results_bw_openmpi.dat"  using 1:2 title "OpenMPI"   with linespoints, \
     "results_bw_mvapich2.dat" using 1:2 title "MVAPICH2"  with linespoints



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




