import sympy as smp
from sympy import diff

x, y, t, nu = smp.symbols('x,y,t,nu')

ft = smp.sin(t)
u1 = ft*x*x*(1-x)*(1-x)*2*y*(1-y)*(2*y-1)
u2 = ft*y*y*(1-y)*(1-y)*2*x*(1-x)*(1-2*x)
p = ft*x*(1-x)*y*(1-y)

# Stokes case
rhs1 = diff(u1,t) - nu*(diff(u1,x,x) + diff(u1,y,y)) + diff(p,x)
rhs2 = diff(u2,t) - nu*(diff(u2,x,x) + diff(u2,y,y)) + diff(p,y)

#rhs3 = div u --- should be zero!!
rhs3 = diff(u1,x) + diff(u2,y)

# Advection (u.D)u
ad1 = ( u1*diff(u1,x) + u2*diff(u1,y) )
ad2 = ( u1*diff(u2,x) + u2*diff(u2,y) )

#rhs1 = smp.simplify(smp.simplify(rhs1) - smp.simplify(ad1))
#rhs2 = smp.simplify(rhs2 - ad2)
rhs3 = smp.simplify(rhs3)

print'rhs1 = \n\t%r \t\t\n' % ( rhs1 - ad1 )
print'rhs2 = \n\t%r \t\t\n' % ( rhs2 - ad2 )
print'rhs3 = \n\t%r \t\t\n' % rhs3

# regexp for replace ** by pow \([xy*+()\[0-9\]-]\)\*\*\([0-9]\)/pow(\1,\2)/g
-nu*(-8*pow(x,2)*y*pow(-x + 1,2)*sin(t) + 4*pow(x,2)*y*(-y + 1)*(2*y - 1)*sin(t) + 8*pow(x,2)*pow(-x + 1,2)*(-y + 1)*sin(t) - 4*pow(x,2)*pow(-x + 1,2)*(2*y - 1)*sin(t) + 8*x*y*(2*x - 2)*(-y + 1)*(2*y - 1)*sin(t) + 4*y*pow(-x + 1,2)*(-y + 1)*(2*y - 1)*sin(t)) - 2*pow(x,2)*y*pow(-x + 1,2)*(-y + 1)*(2*y - 1)*(2*pow(x,2)*y*(2*x - 2)*(-y + 1)*(2*y - 1)*sin(t) + 4*x*y*pow(-x + 1,2)*(-y + 1)*(2*y - 1)*sin(t))*sin(t) + 2*pow(x,2)*y*pow(-x + 1,2)*(-y + 1)*(2*y - 1)*cos(t) - 2*x*pow(y,2)*(-2*x + 1)*(-x + 1)*pow(-y + 1,2)*(4*pow(x,2)*y*pow(-x + 1,2)*(-y + 1)*sin(t) - 2*pow(x,2)*y*pow(-x + 1,2)*(2*y - 1)*sin(t) + 2*pow(x,2)*pow(-x + 1,2)*(-y + 1)*(2*y - 1)*sin(t))*sin(t) - x*y*(-y + 1)*sin(t) + y*(-x + 1)*(-y + 1)*sin(t) 	
