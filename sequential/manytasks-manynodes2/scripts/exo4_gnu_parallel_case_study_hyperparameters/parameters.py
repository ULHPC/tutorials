
# Hyper-parameter space design
stepx=0.1
minx=-2.0
maxx=2.0

stepy=0.1
miny=-2.0
maxy=2.0

parameters=[]

# Grid Search
x=minx
while x<maxx:
    y=miny
    while y<maxy:
        param=(x,y)
        parameters.append( param )
        y+=stepy
    x+=stepx

# parameters content example: [(-2,-2) (-2,-1.8), (-2,-1.6), .... ]

# Convert parameters into string
params_str=[]
for x,y in parameters:
    param_str=str(round(x,3))+","+str(round(y,3))
    params_str.append(param_str)

# params_string content example: ["-2,-2", "-2,-1.8", "-2,-1.6", ....]

output="\n".join(params_str)

# output content example "-2,-2 -2,-1.8, -2,-1.6 ...."

print(output) # the output can be re-redirected to another program

