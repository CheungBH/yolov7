from .rally import RallyChecker
from .server_Chris import ServeChecker


class StrategyManager:
    def __init__(self, check_stage="serve", **kwargs):
        self.check_stage = check_stage
        self.serve_checker = ServeChecker(**kwargs)
        self.rally_checker = RallyChecker(**kwargs)
        self.args = kwargs
        self.previous_position = self.args["serve_position"]

    def update_line(self, lines):
        if self.check_stage == "serve":
            self.serve_checker.update_line(central_y=int((lines[9] + lines[11])//2),
                                           central_x=int((lines[-12] + lines[-10])//2))
        elif self.check_stage == "rally":
            self.rally_checker.update_line(central_y=int((lines[9] + lines[11])//2),
                                           central_x=int((lines[-12] + lines[-10])//2))
        elif self.check_stage == "highlight":
            self.highlight_update_line(lines)

    def process(self, ball_exist, ball_center, humans_box, humans_action,classifier_status, lines,frame,words,human_realbox,ball_realbox):
        if classifier_status == 'highlight':
            self.check_stage = 'serve'
            if hasattr(self.serve_checker, "serve_position"):
                # self.previous_position = self.serve_checker.serve_position
                self.args["serve_position"] = self.serve_checker.serve_position
            self.serve_checker = ServeChecker(**self.args)
            return
        self.update_line(lines)
        if self.check_stage == "serve":
            if self.serve_checker.flag:
                self.check_stage = 'rally'
                self.args["serve_condition"] = "First serve" if self.serve_checker.serve_cnt == 1 else "Second serve"
                self.rally_checker = RallyChecker(**self.args)
                self.update_line(lines)
                self.rally_checker.process(ball_exist, ball_center, humans_box, humans_action,lines,frame,words,human_realbox,ball_realbox)
            else:
                self.serve_checker.process(humans_box, humans_action)
        elif self.check_stage == 'rally':
            self.rally_checker.process(ball_exist, ball_center, humans_box, humans_action, lines, frame, words,
                                       human_realbox, ball_realbox)
            '''
            if  self.rally_checker.end_situation == "Down net":
                self.check_stage = "serve"
                self.previous_position = self.serve_checker.serve_position
                self.args["serve_position"] = self.previous_position
                self.serve_checker = ServeChecker(**self.args)
                self.update_line(lines)
                self.serve_checker.process(humans_box, humans_action)
            else:
                self.update_line(lines)
                self.rally_checker.process(ball_exist, ball_center, humans_box, humans_action,lines,frame,words,human_realbox,ball_realbox)
            '''

    def get_box(self):
        if self.check_stage == "serve":
            return self.serve_checker.get_box()
        else:
            return self.rally_checker.get_box()

    def get_ball(self):
        if self.check_stage == "serve":
            return self.serve_checker.get_ball()
        else:
            return self.rally_checker.get_ball()

    def visualize_strategies(self, frame):
        if self.check_stage == "serve":
            self.serve_checker.visualize(frame)
        else:
            self.rally_checker.visualize(frame)

    # def visualize_others(self, frame, court_detector, topview):
    #     court_detector.visualize(frame)
    #     player_bv, _, speed, heatmap = topview.visualize()
    #     return frame, player_bv, speed, heatmap




