import unittest
from agent import QLearner, SarsaLearner, Teacher
from game import Game, GameLearning

class TestGameAndAgents(unittest.TestCase):

    def setUp(self):
        # use epsilon = 0 so that actions are deterministic
        # and therefore testable
        self.q_agent = QLearner(0.5,0.9,0)
        self.s_agent = SarsaLearner(0.5,0.9,0)
        # test deterministic teacher
        self.teacher = Teacher(1)

        self.game = Game(self.q_agent)

    def testGame(self):
        self.game.computerMove((1,1))
        self.assertEqual(self.game.board[1][1],'O')
        with self.assertRaises(IndexError):
            self.game.computerMove((3,3))
        self.assertFalse(self.game.checkForWin('O'))
        self.assertFalse(self.game.checkForDraw())
        self.assertEqual(self.game.checkForEnd('O'),-1)
        self.game.computerMove((0,0)); self.game.computerMove((2,2))
        self.assertTrue(self.game.checkForWin('O'))
        self.assertFalse(self.game.checkForDraw())
        self.assertEqual(self.game.checkForEnd('O'),1)

    def testLearningAgents(self):
        a1 = self.q_agent.get_action('---------')
        self.assertEqual(a1,(0,0))
        a2 = self.q_agent.get_action('O--------')
        self.assertEqual(a2,(1,0))
        self.q_agent.update('---------','----0X---',(1,1),1)
        self.s_agent.update('---------','----0X---',(1,1),(2,1),-1)
        self.assertEqual(self.q_agent.Q[(1,1)]['---------'],0.5)
        self.assertEqual(self.s_agent.Q[(1,1)]['---------'],-0.5)

    def testTeachingAgent(self):
        board1 = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
        board2 = [['X', '-', 'O'], ['O', 'X', '-'], ['-', '-', '-']]
        # should be center
        self.assertEqual(self.teacher.makeMove(board1),(1,1))
        # should be corner for win
        self.assertEqual(self.teacher.makeMove(board2),(2,2))


if __name__ == '__main__':
    unittest.main()