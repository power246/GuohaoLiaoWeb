def box1_d(pos):
    global box1
    box1=box(pos,50,50)
    box1.color=Color('red')

def box2_d(f1,g1):
    global box2
    box2=box((f1,g1),50,50)
    box2.color=Color('blue')

def box4_d(f2,g2):
    global box4
    box4=box((f2,g2),50,50)
    box4.color=Color('blue')

def box5_d(f3,g3):
    global box5
    box5=box((f3,g3),50,50)
    box5.color=Color('blue')

def box3_d(pos):
    global box3
    box3=cosmetic_box(pos,50,50)
    box3.color=Color('yellow')


def box_move():
    global f,ball1,g,m,n,f1,g1,f2,g2,f3,g3,box2,box4,box5,h1,h2,h3,h
    if f==f1+25 and g==g1+25:
        deactivate(box2)
        f1=f1+m
        g1=g1+n
        box2_d(f1,g1)
        deactivate(ball1)
        ball1=ball((f,g),25)
        ball1.color=Color('green')
    elif f==f2+25 and g==g2+25:
        deactivate(box4)
        f2=f2+m
        g2=g2+n
        box4_d(f2,g2)
        deactivate(ball1)
        ball1=ball((f,g),25)
        ball1.color=Color('green')
    elif f==f3+25 and g==g3+25:
        deactivate(box5)
        f3=f3+m
        g3=g3+n
        box5_d(f3,g3)
        deactivate(ball1)
        ball1=ball((f,g),25)
        ball1.color=Color('green')
    else:
        deactivate(ball1)
        ball1=ball((f,g),25)
        ball1.color=Color('green')
    for i in line:
        h=(f-25,g-25)
        h1=(f1,g1)
        h2=(f2,g2)
        h3=(f3,g3)
        if h1==i or h1==h2 or h1==h3:
            deactivate(box2)
            f1=f1-m
            g1=g1-n
            box2_d(f1,g1)
            deactivate(ball1)
            g=g-n
            f=f-m
            ball1=ball((f,g),25)
            ball1.color=Color('green')
        elif h2==i or h2==h1 or h2==h3:
            deactivate(box4)
            deactivate(ball1)
            f2=f2-m
            g2=g2-n
            box4_d(f2,g2)
            g=g-n
            f=f-m
            ball1=ball((f,g),25)
            ball1.color=Color('green')
        elif h3==i or h3==h1 or h3==h2:
            deactivate(box5)
            deactivate(ball1)
            f3=f3-m
            g3=g3-n
            box5_d(f3,g3)
            g=g-n
            f=f-m
            ball1=ball((f,g),25)
            ball1.color=Color('green')
        elif h==i:
            deactivate(ball1)
            g=g-n
            f=f-m
            ball1=ball((f,g),25)
            ball1.color=Color('green')
    m=0
    n=0
    box_finish()
    
def box_finish():
    if h2==(300,200) and h1==(300,250) and h3==(300,300):
        text1=static_text_with_font((50,150),'YOU WIN!', 'Segoe Script',48) 
        text1.color=Color('black')

def balla_observer(keys):
    global f,ball1,g,m,n,f1,g1,box2
    if constants.K_a in keys:
        m=-50
        f=f+m
        g=g+n
        box_move()
    if constants.K_d in keys:
        m=50
        f=f+m
        g=g+n
        box_move()
    if constants.K_s in keys:
        n=50
        f=f+m
        g=g+n
        box_move()
    if constants.K_w in keys:
        n=-50
        f=f+m
        g=g+n
        box_move()
        
from pyphysicssandbox import*
window('2333',400,400)
gravity(0,0)
f=275
g=75
f1=200
g1=100
f2=250
g2=250
f3=100
g3=300
h1=0
h2=0
h3=0
m=0
n=0
line=[(50,50),(50,100),(50,150),(0,150),(100,50),(150,50),(150,0),(200,0),(250,0),(300,0),
      (300,50),(300,100),(300,150),(350,150),(250,150),(250,200),(350,250),(350,300),
      (350,200),(350,350),(300,350),(250,350),(200,350),(150,350),(100,350),(50,350),
      (0,350),(0,300),(0,250),(0,200),(150,200),(150,150),(100,250)]
finish=[(300,200),(300,250),(300,300)]
ball1=ball((f,g),25)
ball1.color=Color('green')

for i in line:
    box1_d(i)

for x in finish:
    box3_d(x)

box2_d(f1,g1)
box4_d(f2,g2)
box5_d(f3,g3)


add_observer(balla_observer)


run()

'''
power胖丸
'''

