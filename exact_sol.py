import sympy as smp
from sympy import diff

x, y, t, nu = smp.symbols('x,y,t,nu')

ft = smp.sin(t)
u1 = ft*x*x*(1-x)*(1-x)*2*y*(1-y)*(2*y-1)
u2 = ft*y*y*(1-y)*(1-y)*2*x*(1-x)*(1-2*x)
p = ft*x*(1-x)*y*(1-y)

#Stokes case
rhs1 = diff(u1,t)-nu*( diff(u1,x,x)+diff(u1,y,y) )+diff(p,x)
rhs2 = diff(u2,t)-nu*( diff(u2,x,x)+diff(u2,y,y) )+diff(p,y)

#rhs3 = div u --- should be zero!!
rhs3 = diff(u1,x)+diff(u2,y)

rhs1 = smp.simplify(rhs1)
rhs2 = smp.simplify(rhs2)
rhs3 = smp.simplify(rhs3)

print'rhs1 = \n\t%r \t\t\n' % rhs1
print'rhs2 = \n\t%r \t\t\n' % rhs2
print'rhs3 = \n\t%r \t\t\n' % rhs3
