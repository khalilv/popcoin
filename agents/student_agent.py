
# Student agent: Add your own agent here
from copy import deepcopy
import time
import numpy as np 
from agents.agent import Agent
from store import register_agent


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.turn_number = 0
        self.root = None
        self.autoplay = True
        self.root_number_of_moves = None
        self.board_size = None

    def step(self, chess_board, my_pos, adv_pos, max_step):
        self.turn_number += 1
        if self.turn_number == 1: #first move - initialization    
            self.board_size = chess_board.shape[0]    
            timeout = time.time() + 29.5   #29.5 sec from now
            self.root = mcts_node(chess_board, my_pos, adv_pos, max_step, self.neighboring_moves(chess_board, my_pos, max_step, adv_pos), True, False)
            result = self.simulate(self.root)
            self.backpropogate(self.root,result)
        else: #all subsequent moves 
            timeout = time.time() + 1.95  #1.95 sec from now
            child = self.find_expanded_child(self.root, chess_board, adv_pos) 
            if child: #we have already expanded opponents move
                self.root = child
                child.parent = None
                child.parent_action = None
            else: #havent seen opponents move 
                self.root = mcts_node(chess_board, my_pos, adv_pos, max_step, self.neighboring_moves(chess_board, my_pos, max_step, adv_pos), True, False)
                result = self.simulate(self.root)
                self.backpropogate(self.root, result)
        self.root_number_of_moves = self.root.number_of_moves
        while time.time() < timeout: #mcts loop
            node_to_expand = self.select(self.root)
            if node_to_expand: 
                new_child = self.expand(node_to_expand)
                result = self.simulate(new_child)
                self.backpropogate(new_child, result)
            else: #tree solved
                break
        avg_visits = self.get_avgerage_visits(self.root)
        best_child=self.get_max_child(self.root, avg_visits)        
        x,y,dir = best_child.parent_action
        self.root = best_child
        return (x,y),dir

    #check if opponents move has already been expanded 
    #returns node corresponding to current game state if it exists, None if not 
    def find_expanded_child(self, node, chess_board, adv_pos):
        for child in node.children: 
            x,y,dir = child.parent_action
            adv_x, adv_y = adv_pos
            if chess_board[x,y,dir] and adv_x == x and adv_y == y:
                return child
        return None

    #select node to expand next 
    def select(self, node):
        if not node: #edge case - tree has been fully expanded 
            return None
        if len(node.children) == 0: 
            return node
        elif len(node.children) == len(node.possible_moves): #no moves left to explore 
            max_child = None
            max_child_value = None
            for child in node.children:
                if not child.fully_expanded: 
                    child_value = self.evaluate(child)
                    if not max_child or child_value > max_child_value: 
                        max_child_value = child_value
                        max_child = child
            if not max_child: #children have all been fully expanded 
                node.fully_expanded = True #set this node to be fully expanded
                return self.select(node.parent) #run select on parent 
            else: 
                return self.select(max_child) #run select on best child 
        else: 
            return node #explore new move from this node
    
    #expand a node by adding a new child to it
    def expand(self, node):
        x,y,dir = node.possible_moves[node.next_move_index] #get next move to explore 
        board_copy = deepcopy(node.board) #copy board 
        self.place_barrier(board_copy,x,y,dir) #place barrier according to move
        child = mcts_node(board_copy, node.adv_pos, (x,y), node.max_step, self.neighboring_moves(board_copy, node.adv_pos, node.max_step, (x,y)), not node.max_node, False, node, (x,y,dir)) #add new mcts node
        node.children.append(child) #set it as a child of the current node 
        node.next_move_index += 1 #increment move count
        return child #return new node
                
    #simulates a game till the game is over 
    #returns final score 
    def simulate(self, node): 
        chess_board_copy = deepcopy(node.board) #create a copy of the chess board 
        my_pos = node.my_pos
        adv_pos = node.adv_pos
        max_step = node.max_step
        adv_turn = False
        is_endgame, my_points, adv_points = self.is_endgame(chess_board_copy, my_pos, adv_pos) #check for endgame
        if is_endgame: #terminal node
            node.fully_expanded = True #set node to be fully expanded
            if my_points > adv_points:
                return 1 #win
            elif my_points == adv_points:
                return 0.5 #tie
            else: 
                return 0 #loss
        while True: #still moves to be played
            is_endgame, my_points, adv_points = self.is_endgame(chess_board_copy, my_pos, adv_pos) #check for endgame
            if is_endgame: #if the game is over
                if my_points > adv_points:
                    return 1 #win
                elif my_points == adv_points:
                    return 0.5 #tie
                else: 
                    return 0 #loss
            else: #game is not over
                if (adv_turn): #adversary turn 
                    adv_x,adv_y,adv_dir = self.random_move(chess_board_copy, adv_pos, my_pos, max_step) #compute random adversary move
                    self.place_barrier(chess_board_copy, adv_x, adv_y, adv_dir)
                    adv_pos = (adv_x, adv_y) #update position 
                    adv_turn = False #switch turns
                else: #my turn
                    x,y,dir = self.random_move(chess_board_copy, my_pos, adv_pos, max_step) #compute random move
                    self.place_barrier(chess_board_copy, x, y, dir)
                    my_pos = (x, y) #update position
                    adv_turn = True #switch turns 

    #update win/visit count of all parent nodes
    def backpropogate(self, node, result):
        node.number_of_wins += 1 - result #1 if you won, 0 if opponent won, 0.5 if tie 
        node.number_of_visits += 1 #increment number of visits      
        cur = node
        while 1: 
            parent = cur.parent
            if not parent: break #at the parent of the root so break  
            parent.number_of_visits += 1 #increment number of visits 
            if parent.max_node == node.max_node: #your node
                parent.number_of_wins += 1 - result #1 if you won, 0 if opponent won, 0.5 if tie 
            else: #opponents node
                parent.number_of_wins += result #0 if you won, 1 if opponent won, 0.5 if tie 
            cur = parent #backpropgate

    #check to see if game is over 
    #returns (is_endgame, my_points, adv_points)
    def is_endgame(self, chess_board, my_pos, adv_pos):
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)
        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]
        def union(pos1, pos2):
            father[pos1] = pos2
        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)
        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        else: 
            return True, p0_score, p1_score


    #place barrier at given position and direction 
    def place_barrier(self, chess_board, x, y, dir): 
        chess_board[x,y,dir] = True
        move = self.moves[dir]
        chess_board[x + move[0], y + move[1], self.opposites[dir]] = True


    #executed when mcts loop is done
    #from the root, return the best child according to an evaluation function
    def get_max_child(self, node, average_visits):
        max_child = None
        max_child_value = None
        for child in node.children: 
            if child.fully_expanded and (child.number_of_wins == child.number_of_visits): #winning move
                return child
            elif child.fully_expanded and child.number_of_wins == 0: #losing move
                continue
            else:
                child_value = self.final_eval(child, average_visits) #evaluate child 
            if not max_child or child_value > max_child_value: #save child with best evaluation score 
                max_child_value = child_value
                max_child = child
        if not max_child: #no children that don't immediately cause a loss 
            max_child = node.children[0] 
        return max_child #return best child 

    #evaluate a node
    #return a numerical value 
    def evaluate(self, node): 
        match self.board_size: #Varying weights of wins/visits value vs. heuristics and varying uct values for each board size
            case 6:
                if self.turn_number == 1: #first move
                    return (5/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(0.1*np.log(node.parent.number_of_visits/node.number_of_visits)) - (2/8)*node.normalized_number_of_moves - (1/8)*node.normalized_distance_to_adv
                else:
                    return (4/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(0.15*np.log(node.parent.number_of_visits/node.number_of_visits)) - (2.5/8)*node.normalized_number_of_moves - (1.5/8)*node.normalized_distance_to_adv
            case 7: #7x7
                if self.turn_number == 1:
                    return (4/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(0.25*np.log(node.parent.number_of_visits/node.number_of_visits)) - (2.5/8)*node.normalized_number_of_moves - (1.5/8)*node.normalized_distance_to_adv
                else :
                    return (3/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(0.75*np.log(node.parent.number_of_visits/node.number_of_visits)) - (3/8)*node.normalized_number_of_moves - (2/8)*node.normalized_distance_to_adv       
            case 8: #8x8
                if self.turn_number == 1: 
                    return (3/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(0.2*np.log(node.parent.number_of_visits/node.number_of_visits)) - (3/8)*node.normalized_number_of_moves - (2/8)*node.normalized_distance_to_adv
                else:
                    return (3/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(1*np.log(node.parent.number_of_visits/node.number_of_visits)) - (3/8)*node.normalized_number_of_moves - (2/8)*node.normalized_distance_to_adv
            case 9: #9x9
                if self.turn_number == 1: 
                    return (3/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(0.3*np.log(node.parent.number_of_visits/node.number_of_visits))- (1/2)*node.normalized_number_of_moves - (1/8)*node.normalized_distance_to_adv
                else: #The uct value is dependent upon how many possible moves there are from the current position.  Many possible moves leads to less simulations per node
                    c=(self.root_number_of_moves)/(node.max_number_of_moves[node.max_step])
                    return 1/(16*c)*(node.number_of_wins/node.number_of_visits) + np.sqrt(c*np.log(node.parent.number_of_visits/node.number_of_visits)) - (1/2)*node.normalized_number_of_moves - (1/8)*node.normalized_distance_to_adv
            case 10:
                if self.turn_number == 1: #first move
                    return (3/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(0.6*np.log(node.parent.number_of_visits/node.number_of_visits))- (1/2)*node.normalized_number_of_moves - (1/8)*node.normalized_distance_to_adv
                else:
                    c=(self.root_number_of_moves)/(node.max_number_of_moves[node.max_step])
                    return 1/(12*c)*(node.number_of_wins/node.number_of_visits) + np.sqrt(c*np.log(node.parent.number_of_visits/node.number_of_visits)) - (1/2)*node.normalized_number_of_moves - (1/8)*node.normalized_distance_to_adv
            case 11:
                if self.turn_number == 1: #first move
                    return (3/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(0.5*np.log(node.parent.number_of_visits/node.number_of_visits))- (1/2)*node.normalized_number_of_moves - (1/8)*node.normalized_distance_to_adv
                else:
                    c=(self.root_number_of_moves)/(node.max_number_of_moves[node.max_step])
                    return 1/(8*c)*(node.number_of_wins/node.number_of_visits) + np.sqrt(c*np.log(node.parent.number_of_visits/node.number_of_visits)) - (1/2)*node.normalized_number_of_moves - (1/8)*node.normalized_distance_to_adv
            case 12:
                if self.turn_number == 1: #first move
                    return (3/8)*(node.number_of_wins/node.number_of_visits) + np.sqrt(0.5*np.log(node.parent.number_of_visits/node.number_of_visits))- (1/2)*node.normalized_number_of_moves - (1/8)*node.normalized_distance_to_adv
                else:
                    c=(self.root_number_of_moves)/(node.max_number_of_moves[node.max_step])
                    return 1/(4*c)*(node.number_of_wins/node.number_of_visits) + np.sqrt(c*np.log(node.parent.number_of_visits/node.number_of_visits)) - (1/2)*node.normalized_number_of_moves - (1/8)*node.normalized_distance_to_adv
   
    #evaluation function for a node
    #to be run at the end of the mcts loop
    def final_eval(self, node, average_visits):
        if average_visits == 1: #Value the wins/visit value low.  
            return node.normalized_distance_to_adv + node.normalized_number_of_moves + (0.1 * node.number_of_wins)
        elif average_visits < 4: #Relatively low value for wins/visits relative to heuristics
            mcts_factor = 0.2*(((average_visits)-0.9)**0.1)*((node.number_of_wins/node.number_of_visits)-(1/(2*node.number_of_visits)))
            return  mcts_factor - 0.5*node.normalized_distance_to_adv - node.normalized_number_of_moves
        elif average_visits < 20:  #Mid-level value for wins/visits relative to heuristics
            mcts_factor = (((average_visits)-0.9)**0.3)*((node.number_of_wins/node.number_of_visits)-(1/(2*node.number_of_visits)))
            return mcts_factor - 0.2*node.normalized_distance_to_adv - 0.5*node.normalized_number_of_moves
        else:   #If there has been an average of 20 visits to each node, simply return the node which has been visited the most
            return node.number_of_visits
    
    #take a random move
    #returns (x,y,dir)
    def random_move(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)
            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break
        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)
        x,y = my_pos
        return x,y,dir

    #compute list of valid moves 
    def neighboring_moves(self, chess_board, my_pos, max_step, adv_pos):
        possible_moves = []
        state_queue = [(my_pos, 0)]
        visited = {tuple(my_pos)}
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue
                next_pos = (r + move[0], c + move[1])
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        for state in visited: 
            x,y = state
            if not chess_board[x,y,self.dir_map['u']]:
                possible_moves.append((x,y,self.dir_map['u']))
            if not chess_board[x,y,self.dir_map['d']]:
                possible_moves.append((x,y,self.dir_map['d']))
            if not chess_board[x,y,self.dir_map['l']]:
                possible_moves.append((x,y,self.dir_map['l']))
            if not chess_board[x,y,self.dir_map['r']]:
                possible_moves.append((x,y,self.dir_map['r']))
        return possible_moves

    
    #helper function to get average number of moves and average number of visits of a nodes children
    #returns (average_visits, average_moves)
    def get_avgerage_visits(self,node):
        sum_visits = sum(child.number_of_visits for child in node.children)
        num_children = len(node.children)
        return sum_visits/num_children

class mcts_node():
    max_number_of_moves = {1: 20, 2: 52, 3: 100, 4: 164, 5: 244, 6: 336}
    max_distance_to_adv = {5: 8, 6: 10, 7: 12, 8: 14, 9: 16, 10: 18, 11: 20, 12: 22}
    def __init__(self, board, my_pos, adv_pos, max_step, possible_moves, max_node, fully_expanded, parent=None, parent_action=None):
        possible_moves.sort(key=lambda x: abs(adv_pos[0]-x[0]) + abs(adv_pos[1]-x[1]))
        self.board = board
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.number_of_visits = 0
        self.next_move_index = 0
        self.number_of_wins = 0
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.possible_moves = possible_moves
        self.max_node = max_node
        self.fully_expanded = fully_expanded
        self.number_of_moves = len(possible_moves)
        self.normalized_distance_to_adv = (abs(my_pos[0]-adv_pos[0]) + abs(my_pos[1]-adv_pos[1]))/mcts_node.max_distance_to_adv[board.shape[0]]
        self.normalized_number_of_moves = self.number_of_moves/mcts_node.max_number_of_moves[max_step]
        return
