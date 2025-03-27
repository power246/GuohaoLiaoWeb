'''
Project 1
Paul Liao
'''

import random

def is_valid_move(board, row, col, color):
    '''
    is_valid_move is a function to test is the player's input valid or not.
    It has four parameter, (the board, the number of
    row that player wants to input, the number
    of columns that player wants to input, the color of the player)
    The valid one must be unoccupied.
    At least one of its 8 neighbors must be occupied by an
    opponent's color.At least one of the straight lines
    starting from row, col and continuing horizontally,
    vertically or diagonally in any direction must start
    with an opponent’s token and end with a token
    matching the color argument..
    retrun Boolean
    '''
    lst = board[:]
    if row >= len(lst):
        return False
    if col >= len(lst[0]):
        return False
    if lst[row][col] == 1 or lst[row][col] == 2:
        return False
    a, b, c = get_a_string(row, col, lst, color)
    if b == 2:
        return True
    return False


def get_a_string(row, col, lst, color):
    '''
    a represents how many pieces are connected
    b represents end with op_color
    c represents the coordinates of op_color chess
    '''
    op_row = []
    op_col = []
    a = 0
    b = 0
    c = ()
    if color == 1:
        op_color = 2
    if color == 2:
        op_color = 1
    if color == 'white':
        color = 1
        op_color = 2
    if color == 'black':
        color = 2
        op_color = 1
    for i in range(3):
        for ii in range(3):
            if i != 1 or ii != 1:
                if (i+row-1>=0 and i+row-1<len(lst) and ii+col-1>=0
                    and ii+col-1<len(lst[0])):
                    if lst[i+row-1][ii+col-1] == op_color:
                        op_row.append(i+row-1)
                        op_col.append(ii+col-1)
    for x in range(len(op_row)):
        a = 2
        b = 0
        c = ()
        while (row-a*(row-op_row[x])>=0 and row-a*(row-op_row[x])<len(lst)
               and col-a*(col-op_col[x])>=0
               and col-a*(col-op_col[x])<len(lst[0]) and b == 0):
            if lst[row-a*(row-op_row[x])][col-a*(col-op_col[x])] == color:
                b = 2
                c = (op_row[x], op_col[x])
                return a,b,c
            if lst[row-a*(row-op_row[x])][col-a*(col-op_col[x])] == 0:
                b = 1
            a+=1
    return a,b,c



def get_valid_moves(board, color):
    '''
    To find all valid moves in the board.
    return list
    '''
    lst = board[:]
    v_lst = []
    for i in range(len(lst)):
        for ii in range(len(lst[0])):
            if is_valid_move(lst, i, ii, color):
                v_lst.append((i,ii))
    return v_lst


def select_next_play_random(board, color):
    '''
    Computer player with random choice.
    return tuple
    '''
    v_lst = get_valid_moves(board, color)
    num = random.randint(0,len(v_lst)-1)
    return v_lst[num]


def select_next_play_ai(board, color):
    '''
    It will choose the solution with the most flips
    return tuple
    '''
    lst = board[:]
    v_lst = get_valid_moves(board, color)
    n_max = 0
    for u in v_lst:
        row = u[0]
        col = u[1]
        a, b, c = get_a_string(row, col, lst, color)
        if a>n_max:
            n_max = a
            choice = (row, col)
    return choice


def select_next_play_human(board, color):
    '''
    To determine whether human player choice is valid.
    return tuple
    '''
    row = int(input('Select a row: '))
    col = int(input('Select a column: '))
    while not(is_valid_move(board, row, col, color)):
        print('Invalid choice')
        row = int(input('Select a row: '))
        col = int(input('Select a column: '))
    return (row, col)


def get_board_as_string(board):
    '''
    To draw a board.
    '''
    s = ''
    s = s + '  '
    for x in range(len(board[0])):
        s = s + ' '+str(x%10)
    s = s + '\n'
    s = s + '  +'
    for ii in range(len(board[0])):
        s += '-+'
    s += '\n'
    for i in range(len(board)):
        if board[i][0] == 0:
            s += str(i)+' | '
        if board[i][0] == 1:
            s += str(i)+' |'+ '\U000025CB'
        if board[i][0] == 2:
            s += str(i)+' |'+ '\U000025CF'
        for ii in range(len(board[i])-1):
            ii+=1
            if board[i][ii] == 0:
                s += '| '
            if board[i][ii] == 1:
                s += '|'+ '\U000025CB'
            if board[i][ii] == 2:
                s += '|'+ '\U000025CF'
        s += '|' + '\n'
        s += '  +'
        for ii in range(len(board[0])):
            s += '-+'
        s += '\n'
    return s

def set_up_board(width, height):
    '''
    To create a board.
    return list
    '''
    lst = []
    for i in range(height):
        lst.append([])
        for ii in range(width):
            if i == int((height/2))-1 and ii == (int(width/2))-1:
                lst[i].append(1)
            elif i == int((height/2))-1 and ii == int((width/2)):
                lst[i].append(2)
            elif i == int((height/2)) and ii == int((width/2))-1:
                lst[i].append(2)
            elif i == int((height/2)) and ii == int((width/2)):
                lst[i].append(1)
            else:
                lst[i].append(0)
    return lst


def flip(player, board, color):
    '''
    Flip the color of the chess pieces connected in a straight line.
    '''
    p = player
    board[p[0]][p[1]] = color
    a, b, c = get_a_string(p[0], p[1], board, color)
    for x in range(1, a):
        board[x*(c[0]-p[0])+p[0]][x*(c[1]-p[1])+p[1]] = color



def human_vs_random():
    '''
    The function of human vs. random AI
    '''
    board = set_up_board(8, 8)
    u = 0
    w_n = 0
    b_n = 0
    while u == 0:
        if get_valid_moves(board, 'white') != []:
            print("Player 1's Turn")
            print(get_board_as_string(board))
            hum = select_next_play_human(board, 'white')
            board[hum[0]][hum[1]] = 1
            flip(hum, board, 1)
        else:
            u = 1
        if get_valid_moves(board, 'black') != [] and u != 1:
            print("Player 2's Turn")
            print(get_board_as_string(board))
            ran = select_next_play_random(board, 'black')
            board[ran[0]][ran[1]] = 2
            flip(ran, board, 2)
        else:
            u = 1
    for i in range(len(board)):
        for ii in board[i]:
            if ii == 1:
                w_n += 1
            if ii == 2:
                b_n += 1
    print('Final Board State')
    print(get_board_as_string(board))
    if w_n>b_n:
        print('Player 1 Wins')
        return 1
    elif w_n<b_n:
        print('Player 2 Wins')
        return 2
    else:
        print('Even')
        return 0


def ai_vs_random():
    '''
    The function of AI vs. random AI
    '''
    board = set_up_board(4, 4)
    u = 0
    w_n = 0
    b_n = 0
    while u == 0:
        if get_valid_moves(board, 'white') != []:
            print("Player 1's Turn")
            print(get_board_as_string(board))
            ai = select_next_play_ai(board, 'white')
            board[ai[0]][ai[1]] = 1
            flip(ai, board, 1)
        else:
            u = 1
        if get_valid_moves(board, 'black') != [] and u != 1:
            print("Player 2's Turn")
            print(get_board_as_string(board))
            ran = select_next_play_random(board, 'black')
            board[ran[0]][ran[1]] = 2
            flip(ran, board, 2)
        else:
            u = 1
    for i in range(len(board)):
        for ii in board[i]:
            if ii == 1:
                w_n += 1
            if ii == 2:
                b_n += 1
    print('Final Board State')
    print(get_board_as_string(board))
    if w_n>b_n:
        print('Player 1 Wins')
        return 1
    elif w_n<b_n:
        print('Player 2 Wins')
        return 2
    else:
        print('Even')
        return 0


def random_vs_random():
    '''
    The function of random AI vs. random AI
    '''
    board = set_up_board(8, 8)
    u = 0
    w_n = 0
    b_n = 0
    while u == 0:
        if get_valid_moves(board, 'white') != []:
            print("Player 1's Turn")
            print(get_board_as_string(board))
            ran_1 = select_next_play_random(board, 'white')
            board[ran_1[0]][ran_1[1]] = 1
            flip(ran_1, board, 1)
        else:
            u = 1
        if get_valid_moves(board, 'black') != [] and u != 1:
            print("Player 2's Turn")
            print(get_board_as_string(board))
            ran_2 = select_next_play_random(board, 'black')
            board[ran_2[0]][ran_2[1]] = 2
            flip(ran_2, board, 2)
        else:
            u = 1
    for i in range(len(board)):
        for ii in board[i]:
            if ii == 1:
                w_n += 1
            if ii == 2:
                b_n += 1
    print('Final Board State')
    print(get_board_as_string(board))
    if w_n>b_n:
        print('Player 1 Wins')
        return 1
    elif w_n<b_n:
        print('Player 2 Wins')
        return 2
    else:
        print('Even')
        return 0


ai_vs_random()
