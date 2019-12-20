MUTATION_RATE = 0.5
from game.board import  OthelloGame
from game.board import  Board
from game.board import display
from  Arena import Arena
import copy
import numpy as np
import random
from tqdm import tqdm
class GeneticAlgorithm():
    def __init__(self,population_size,individual_size,iter_size):
        self.population_size = population_size
        self.individual_size = individual_size
        self.iter_size = iter_size
        self.popul = [
            [5, 5, -10, -4, 4, 4],
            [1, 2, -1, -1, 1, 1],
            [13, 5, -9, -3, 3, 3],
            [15, 3, -6, -6, 4, 2],
            [10, 8, -7, -4, 3, 5]
        ]
        self.fit_score = []
        self.negative = [2,3]
        self.game = OthelloGame()
        #self.Arena = Arena(game=self.game,player1=GAAI,player2=GAAI,display=None)
    def calculate_fitness(self):
        res1 = 2
        res2 = 0
        self.fit_score = [0] * self.population_size
        train_size = self.population_size**2
        bar = tqdm(total=train_size)
        bar.set_description("battle process")
        for x in range(self.population_size):
            for y in range(self.population_size):
                if x != y:
                    p_count,o_count = self.chess_game(weight1=self.popul[x],weight2=self.popul[y])
                    if p_count>o_count:
                        res1 = 2
                        res2 = 0
                    elif p_count==o_count:
                        res1 =1
                        res2 =1
                    else:
                        res1 = 0
                        res2 = 2
                self.fit_score[x] += res1
                self.fit_score[y] += res2
                bar.update(1)
        bar.close()
    def cross_over(self,individual1,individual2):
        index = random.randint(0,self.individual_size-1)
        new_individual1 = individual1[:index] + individual2[index:]
        new_individual2 = individual2[:index] + individual1[index:]
        return new_individual1,new_individual2
    def mutation_operation(self,individual):
        for i in range(self.individual_size):
            p = np.random.random()
            if p <= MUTATION_RATE:
                weight = random.uniform(0.7,1.3)
                individual[i] = individual[i] * weight
        return individual

    def genetic_rate(self):
        total_sum = 0
        # 计算适应度总和
        length1 = len(self.fit_score)
        for i in range(length1):
            total_sum += self.fit_score[i]
        gene_rate = []
        # 计算每一个个体的遗传概率
        for i in range(length1):
            tmp = (float)(self.fit_score[i]) / total_sum
            gene_rate.append(tmp)

        return gene_rate
    def train(self):

        for i in range(self.iter_size):
            self.calculate_fitness()
            print('iter : ',i+1,' ',end='')

            max_fit_score = max(self.fit_score)
            max_fit_index = self.fit_score.index(max_fit_score)
            new_popul = []
            new_popul.append(self.popul.pop(max_fit_index))
            self.fit_score.pop(max_fit_index)
            length1 = len(self.fit_score)
            gene_rate = self.genetic_rate()
            for j in range((int)(length1/2)):
                select_choice = np.random.choice(length1,2,False,gene_rate)
                individual1,individual2 = self.cross_over(self.popul[select_choice[0]],self.popul[select_choice[1]])
                individual1 = self.mutation_operation(individual1)
                individual2 = self.mutation_operation(individual2)
                new_popul.append(individual1)
                new_popul.append(individual2)
            self.popul = new_popul
            print('current optimal train weight: ',self.popul[max_fit_index])
            with open('best_gene.txt','w') as f:
                for x in self.popul[max_fit_index]:
                    print(x)
                    f.write(str(x))
                    f.write(' ')
        max_fit_score = max(self.fit_score)
        max_fit_index = self.fit_score.index(max_fit_score)

        print('train weight: ',self.popul[max_fit_index])

    def chess_game(self,weight1,weight2):
        board = Board()
        game = OthelloGame()
        player1 = GAAI(game, board,weight1)
        player2 = GAAI(game, board,weight2)
        curr_player = player1
        curr_color = 1
        game_process = tqdm(total=64)
        game_process.set_description('game process: ')
        while True:
            Opponent = GAAI(game,board,curr_player.ele_weights)
            _,next_step = curr_player.alpha_beta(Opponent,4,-1*float('inf'),float('inf'))
            #print(next_step)
            if next_step is None:
                break
                #if game.getGameEnded(board.pieces,curr_color):
                #    break
            else:
                board.execute_move(move=next_step, color=1)
                #display(np.copy(board.pieces))
                if game.getGameEnded(board.pieces,curr_color):
                    game_process.close()
                    break
                game_process.update(1)
                curr_color *= -1
                for i in range(8):
                    for j in range(8):
                        board.pieces[i][j] *= -1
                if curr_color == 1:
                    curr_player = player1
                else:
                    curr_player = player2
        for i in range(8):
            for j in range(8):
                board.pieces[i][j] *= curr_color
        p_count,o_count,e_count = board.countAll(1)
        return (p_count,o_count)
class GAAI():
    def __init__(self,game,board,ele_weights,player_num=1):
        self.board = board
        self.game = game
        _,self.size = game.getBoardSize()
        self.player_num = player_num
        self.ele_weights = ele_weights
        self.pos_weight = [
            [20, -3, 11, 8, 8, 11, -3, 20],
            [-3, -7, -4, 1, 1, -4, -7, -3],
            [11, -4, 2, 2, 2, 2, -4, 11],
            [8, 1, 2, -3, -3, 2, 1, 8],
            [8, 1, 2, -3, -3, 2, 1, 8],
            [11, -4, 2, 2, 2, 2, -4, 11],
            [-3, -7, -4, 1, 1, -4, -7, -3],
            [20, -3, 11, 8, 8, 11, -3, 20]
        ]
        self.pos_imt = [(0, 0), (0, 7), (7, 0), (7, 7)]
        self.iter = {0:80,1:200,2:300,3:300}

    def evaluation_score(self):
        score1 = 0
        score2 = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board.pieces[i][j] == -self.player_num:
                    score2 += 1
                elif self.board.pieces[i][j] == self.player_num:
                    score1 += 1
        return score1 - score2
    def evaluation_weight(self):
        score = 0
        num1 = 0
        num2 = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board.pieces[i][j] == self.player_num:
                    num1 += 1
                    score += self.pos_weight[i][j]
                elif self.board.pieces[i][j] == self.player_num:
                    score -= self.pos_weight[i][j]
                    num2 += 1
        return [score,num1-num2]
    def evaluation_Comprehensive(self):
        pos_score,num_score = self.evaluation_weight()[0],self.evaluation_weight()[1]
        valid_pos = self.board.get_legal_moves(color=self.player_num)
        action_score = len(valid_pos)
        my_stabilizer = 0
        opp_stabilizer = 0
        if self.board.pieces[0][0] == self.player_num:
            my_stabilizer += 1
        elif self.board.pieces[0][0] == -self.player_num:
            opp_stabilizer += 1
        if self.board.pieces[0][self.size - 1] == self.player_num:
            my_stabilizer += 1
        elif self.board.pieces[0][self.size - 1] == -self.player_num:
            opp_stabilizer += 1
        if self.board.pieces[self.size - 1][0] == self.player_num:
            my_stabilizer += 1
        elif self.board.pieces[self.size - 1][0] == -self.player_num:
            opp_stabilizer += 1
        if self.board.pieces[self.size - 1][self.size - 1] == self.player_num:
            my_stabilizer += 1
        elif self.board.pieces[self.size - 1][self.size - 1] == -self.player_num:
            opp_stabilizer += 1

        weights = [10, 10, 10, 100]
        total_score = pos_score * weights[0] + num_score * weights[1] + action_score * weights[2] + (my_stabilizer - opp_stabilizer) * weights[3]
        if self.board.pieces[1][1] == self.player_num:
            total_score -= 32
        elif self.board.pieces[1][1] == -self.player_num:
            total_score += 32
        if self.board.pieces[1][4] == self.player_num:
            total_score -= 32
        elif self.board.pieces[1][4] == -self.player_num:
            total_score += 32
        if self.board.pieces[4][1] == self.player_num:
            total_score -= 32
        elif self.board.pieces[4][1] == -self.player_num:
            total_score += 32
        if self.board.pieces[4][4] == self.player_num:
            total_score -= 32
        elif self.board.pieces[4][4] == -self.player_num:
            total_score += 32
        return total_score
    def evaluation_best(self):
        chess_num = 0
        out_corner, out_edge, inner_corner, inner_edge = 0, 0, 0, 0
        for i in range(self.size):
            for j in range(self.size):
                if self.board.pieces[i][j]==0:
                    continue
                amount = 1 if self.board.pieces[i][j] == self.player_num else -1
                chess_num += amount
                if i == 0 or j == 0 or i == self.size - 1 or j == self.size - 1:
                    if (i == 0 and j == 0) or (i == 0 and j == self.size - 1) or (i == self.size - 1 and j == 0) or (
                            i == self.size - 1 and j == self.size - 1):
                        out_corner += amount
                    else:
                        out_edge += amount

                elif i == 1 or i == self.size - 2 and (j > 1 and j < self.size - 2):
                    x = self.size - 1 if i == self.size - 2 else 0
                    for k in range(j - 1, j + 2):
                        if self.board.pieces[x][k] == 0:
                            inner_edge += amount
                elif j == 1 or j == self.size - 2 and (i > 1 and i < self.size - 2):
                    y = self.size - 1 if j == self.size - 2 else 0
                    for k in range(i - 1, i + 2):
                        if self.board.pieces[k][y] ==0:
                            inner_edge += amount
                elif j == 1 and i == 1:
                    if self.board.pieces[0][0] == 0:
                        inner_corner += amount
                    for k in range(1, 3):
                        if self.board.pieces[k][0] == 0:
                            inner_edge += amount
                    for k in range(1, 3):
                        if self.board.pieces[0][k] == 0:
                            inner_edge += amount
                elif j == self.size - 2 and i == 1:
                    if self.board.pieces[0][self.size - 1] ==0:
                        inner_corner += amount
                    for k in range(1, 3):
                        if self.board.pieces[k][self.size - 1] == 0:
                            inner_edge += amount
                    for k in range(j - 1, j + 1):
                        if self.board.pieces[0][k] == '.':
                            inner_edge += amount

                            # 内层左下角
                elif j == 1 and i == self.size - 2:
                    if self.board.pieces[self.size - 1][0] == 0:
                        inner_corner += amount
                    for k in range(i - 1, i + 1):
                        if self.board.pieces[k][0] == 0:
                            inner_edge += amount
                    for k in range(1, 3):
                        if self.board.pieces[self.size - 1][k] == 0:
                            inner_edge += amount

                            # 内层右下角
                elif j == self.size - 2 and i == self.size - 2:
                    if self.board.pieces[self.size - 1][self.size - 1] ==0:
                        inner_corner += amount
                    for k in range(i - 1, i + 1):
                        if self.board.pieces[k][self.size - 1] == 0:
                            inner_edge += amount
                    for k in range(j - 1, j + 1):
                        if self.board.pieces[self.size - 1][k] == 0:
                            inner_edge += amount
        valid_position = self.game.getValidMoves(board=self.board,player=self.player_num)
        action_num = len(valid_position)

        final_score = out_corner * self.ele_weights[0] + out_edge * self.ele_weights[1] + inner_corner * \
                                                                                          self.ele_weights[2]
        + inner_edge * self.ele_weights[3] + chess_num * self.ele_weights[4] + action_num * self.ele_weights[5]
        return  final_score
    def alpha_beta(self,opponent,depth,my_score,opponent_score):
        if (depth==0):
            score = self.evaluation_best()
            return score,None
        #cannonicalB = self.game.getCanonicalForm(self.board.pieces,1)
        #valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)
        #print(self.board.pieces)
        valid_position = self.board.get_legal_moves2(1)
        action_num = len(valid_position)
        if (action_num==0):
            score = self.evaluation_best()
            return score,None
        best_score = -999999
        best_action_Next = None
        for a in valid_position:
            #execute_move
            b = Board(self.size)
            for i in range(self.size):
                for j in range(self.size):
                    b.pieces[i][j] = self.board.pieces[i][j]

            self.board.execute_move(a,1)
            #display(np.copy(self.board.pieces))#test
            '''
                      try:
                self.board.execute_move(a,1)
            except:
                print(a)
            
          '''

            for i in range(self.size):
                for j in range(self.size):
                    opponent.board.pieces[i][j] = self.board.pieces[i][j]*-1
            #next_state = self.board.execute_move(move=a,color=self.player_num)
            score,next_step = opponent.alpha_beta(self,depth-1,-opponent_score,-my_score)
            #self.board.pieces = np.copy(b.pieces)
            for i in range(self.size):
                for j in range(self.size):
                    self.board.pieces[i][j] = b.pieces[i][j]
            score = -1*score
            if score > best_score:
                best_score = score
                best_action_Next = a
            if best_score>opponent_score:
                break
        return best_score,best_action_Next

if __name__ == '__main__':
    iter_size = 5
    population_size = 5
    individuals_size = 6
    ga = GeneticAlgorithm(population_size, individuals_size, iter_size)
    ga.train()